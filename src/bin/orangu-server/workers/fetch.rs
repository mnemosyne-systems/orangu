// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! A worker that downloads only what it runs (`[workers].download =
//! range`).
//!
//! A worker runs a range of layers, and building the model reads only its
//! vectors — norms, biases — while its matrices are read when a layer runs.
//! So a worker with no copy of the model fetches, at start, every tensor
//! outside the layer blocks (the embedding and the head, which a Gemma 4
//! worker reads for its per-layer inputs, are among them) and every block's
//! vectors; the blocks themselves when a parent assigns them, before the
//! model check reads them. The file is sparse: what was not fetched is a
//! hole, never read.
//!
//! The bytes come from the nodes of the channel that hold the whole file
//! ([`super::share`]) before the Hub: the node that claimed this one first,
//! then the others whose probe names the model, least busy first, then the
//! Hub. So a channel with no internet needs one node with the model on
//! disk. What a peer sends is checked against the fingerprint its probe
//! gave: the header against the model's layout hash before anything is
//! built from it, every layer's vectors and weight samples against its
//! content hash before the model is built, and each assigned range against
//! the parent's hash before it runs. Bytes that fail are fetched again from
//! the next source, and the source that sent them is not asked again.
//!
//! Such a node cannot serve on its own, so its own API stays off.

use super::channel::{self, Found};
use super::identity;
use super::protocol::FEATURE_FETCH;
use super::share::PeerSource;
use super::transport::Tls;
use crate::engine::loader::LoadedModel;
use anyhow::{Context, Result, bail};
use orangu::model_download::{FileOffer, PartialModel, RangeSource};
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long a node with no copy of its model waits at start for a node of
/// its channel to serve it, when the Hub cannot be reached either.
const WAIT_FOR_A_SOURCE: Duration = Duration::from_secs(600);

/// How often it looks again meanwhile.
const LOOK_AGAIN: Duration = Duration::from_secs(5);

/// How many times bytes that fail a check are fetched again before the
/// check's failure stands.
pub const REFETCHES: usize = 3;

static PARTIAL: OnceLock<Arc<PartialModel>> = OnceLock::new();

/// Makes `partial` the model this process runs.
pub fn set(partial: Arc<PartialModel>) {
    let _ = PARTIAL.set(partial);
}

/// The partial model this process runs, if it runs one.
pub fn partial() -> Option<&'static Arc<PartialModel>> {
    PARTIAL.get()
}

/// Whether this process runs a model it has not fetched whole.
pub fn incomplete() -> bool {
    partial().is_some_and(|p| !p.complete())
}

/// What building the model reads: every tensor outside the layer blocks,
/// and every block's vectors.
pub fn needed_to_build(name: &str, dims: &[u64]) -> bool {
    crate::engine::loader::block_index(name).is_none() || dims.len() <= 1
}

/// How a node fetching its model reaches the rest of its channel before
/// its own node has started: what [`super::node::Node`] scans with.
pub struct Reach {
    /// `[workers].name`; `None` asks no one but the Hub.
    pub channel: Option<String>,
    pub secret: Option<String>,
    pub tls: Option<Tls>,
    pub scan: super::node::Scan,
}

/// Whether the node `found` describes serves the model `spec` names, whole,
/// by range.
fn serves(found: &Found, spec: &str) -> bool {
    found.features & FEATURE_FETCH != 0
        && found.content != [0; 32]
        && channel::names_model(spec, &found.label, &found.quant)
}

/// The nodes of the channel that serve `spec` by range, least busy first,
/// all of one model: the one the least busy serves.
fn offers(reach: &Reach, spec: &str) -> Vec<Found> {
    let Some(name) = reach.channel.as_deref() else {
        return Vec::new();
    };
    let hash = channel::channel_hash(name, reach.secret.as_deref());
    let peers = channel::resolve_peers(&reach.scan.peers);
    let mut found = channel::probe_all(&peers, &hash, reach.tls.as_ref());
    if let Some(networks) = &reach.scan.networks {
        found.extend(channel::scan(
            &channel::scan_hosts(networks),
            reach.scan.port,
            &hash,
            reach.tls.as_ref(),
        ));
    }
    let mut found: Vec<Found> = channel::one_per_node(found, &[])
        .into_iter()
        .filter(|f| serves(f, spec))
        .collect();
    found.sort_by(|a, b| a.load.total_cmp(&b.load).then(a.rtt.cmp(&b.rtt)));
    if let Some(first) = found.first().map(|f| (f.model, f.content)) {
        found.retain(|f| (f.model, f.content) == first);
    }
    found
}

/// Each node's source, kept across fetches: its connection, and what it
/// serves, are learnt once.
static PEERS: OnceLock<Mutex<HashMap<String, Arc<PeerSource>>>> = OnceLock::new();

/// The source for the node `found` describes.
fn peer(found: &Found, secret: Option<&str>, tls: Option<&Tls>) -> Arc<PeerSource> {
    let key = format!("{} ({})", found.name, found.node);
    PEERS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .entry(key.clone())
        .or_insert_with(|| {
            Arc::new(PeerSource::new(
                key,
                found.addr.clone(),
                secret.map(str::to_string),
                tls.cloned(),
            ))
        })
        .clone()
}

/// Opens `spec` in part and fetches what building it reads: from the
/// nodes of the channel that serve it, else from the Hub. With neither, it
/// looks again for [`WAIT_FOR_A_SOURCE`] when it is on a channel.
pub fn open(models_dir: &Path, spec: &str, reach: &Reach) -> Result<Arc<PartialModel>> {
    let started = Instant::now();
    let mut said = false;
    loop {
        let found = offers(reach, spec);
        let opened = match from_channel(models_dir, spec, reach, &found) {
            Some(opened) => opened,
            None => PartialModel::open(models_dir, spec)
                .map(Arc::new)
                .and_then(|partial| {
                    partial.fetch(needed_to_build)?;
                    Ok(partial)
                }),
        };
        match opened {
            Ok(partial) => {
                log::info!(
                    "orangu-server: [workers] download = range: fetched what building {spec} \
                     reads in {:.1} s; each layer range comes when a parent assigns it",
                    started.elapsed().as_secs_f64()
                );
                return Ok(partial);
            }
            Err(error) if reach.channel.is_some() && started.elapsed() < WAIT_FOR_A_SOURCE => {
                if !said {
                    log::warn!(
                        "orangu-server: [workers] download = range: {spec} could not be \
                         fetched ({error:#}); waiting for a node of channel '{}' to serve it",
                        reach.channel.as_deref().unwrap_or_default()
                    );
                    said = true;
                }
                std::thread::sleep(LOOK_AGAIN);
            }
            Err(error) => return Err(error),
        }
    }
}

/// Opens `spec` in part from the channel nodes `found`, with the Hub after
/// them, and fetches what building it reads, checked. `None` when no node
/// answers.
fn from_channel(
    models_dir: &Path,
    spec: &str,
    reach: &Reach,
    found: &[Found],
) -> Option<Result<Arc<PartialModel>>> {
    let mut sources: Vec<Arc<dyn RangeSource>> = Vec::new();
    let mut offer: Option<FileOffer> = None;
    let mut expected = None;
    for f in found {
        let source = peer(f, reach.secret.as_deref(), reach.tls.as_ref());
        let info = match source.info() {
            Ok(info) => info,
            Err(error) => {
                log::warn!(
                    "orangu-server: [workers] {} does not serve its model: {error:#}",
                    f.name
                );
                continue;
            }
        };
        // What its handshake says must be what its probe said, and what
        // the first node serves.
        if info.model != f.model || offer.as_ref().is_some_and(|o| o.size != info.size) {
            continue;
        }
        offer.get_or_insert(FileOffer {
            name: info.file.clone(),
            commit: info.commit.clone(),
            size: info.size,
        });
        expected.get_or_insert((f.model, f.content));
        sources.push(source);
    }
    let offer = offer?;
    let (model, content) = expected?;
    let names: Vec<String> = sources.iter().map(|s| s.key()).collect();
    log::info!(
        "orangu-server: [workers] download = range: fetching {spec} from {}, then the Hub",
        names.join(", ")
    );
    let check = |path: &Path| -> Result<()> {
        let loaded = LoadedModel::open(path)?;
        if identity::header_hash(&loaded) != model {
            bail!("its header is not the model the channel's probe named");
        }
        Ok(())
    };
    Some(
        PartialModel::open_offered(models_dir, spec, &offer, sources, true, &check)
            .map(Arc::new)
            .and_then(|partial| {
                build_checked(&partial, content)?;
                Ok(partial)
            }),
    )
}

/// Fetches what building the model reads, and samples of every layer's
/// weights, and checks them against `content`, the hash a node serving the
/// model gave in its probe. Bytes that fail are fetched again, from the
/// next source.
fn build_checked(partial: &PartialModel, content: [u8; 32]) -> Result<()> {
    for attempt in 0..=REFETCHES {
        partial.fetch(needed_to_build)?;
        let loaded = LoadedModel::open(partial.path())?;
        let n_layer = loaded.config.n_layer;
        let used = partial.fetch_bytes(&identity::sampled_ranges(&loaded, 0..n_layer))?;
        let got = identity::model_identity(&loaded, "", "", 0..n_layer).range_hash;
        if got == content {
            return Ok(());
        }
        // Tensors an earlier run left name no source; the first time they
        // are only fetched again.
        let dropped = partial.forget(needed_to_build, true)?;
        if attempt > 0 || !dropped.is_empty() {
            for key in &used {
                partial.reject(key);
            }
        }
        log::warn!(
            "orangu-server: [workers] the layers' weights fetched do not match the channel's \
             model; fetching them again{}",
            if dropped.is_empty() {
                String::new()
            } else {
                format!(" without {}", dropped.join(", "))
            }
        );
    }
    bail!("the weights fetched never matched the model the channel serves")
}

/// The sources of the channel to fetch a parent's assignment from, in
/// order: the parent `parent` (its node id), then the other nodes `found`
/// that serve the model `model`, least busy first. The Hub, and the nodes
/// found at start, stay behind them.
pub fn prefer(
    found: &[Found],
    parent: Option<&str>,
    model: [u8; 32],
    secret: Option<&str>,
    tls: Option<&Tls>,
) {
    let Some(partial) = partial() else {
        return;
    };
    partial.prefer(
        in_order(found, parent, model)
            .into_iter()
            .map(|f| peer(f, secret, tls) as Arc<dyn RangeSource>)
            .collect(),
    );
}

/// The nodes of `found` that serve the model `model` whole, by range: the
/// parent `parent` first, then the least busy. A node that did not
/// advertise [`FEATURE_FETCH`] is never among them.
fn in_order<'a>(found: &'a [Found], parent: Option<&str>, model: [u8; 32]) -> Vec<&'a Found> {
    let mut serving: Vec<&Found> = found
        .iter()
        .filter(|f| f.features & FEATURE_FETCH != 0 && f.content != [0; 32] && f.model == model)
        .collect();
    serving.sort_by(|a, b| {
        (Some(a.node.as_str()) != parent)
            .cmp(&(Some(b.node.as_str()) != parent))
            .then(a.load.total_cmp(&b.load))
    });
    serving
}

fn in_layers(layers: &Range<usize>) -> impl Fn(&str, &[u64]) -> bool + '_ {
    |name, _| crate::engine::loader::block_index(name).is_some_and(|il| layers.contains(&il))
}

/// Fetches `layers`' tensors, if this process runs a partial model, and
/// has a device place again what it placed of them before: anything that
/// ran them before they were fetched read the file's holes.
pub fn layers(loaded: &LoadedModel, layers: &Range<usize>) -> Result<()> {
    let Some(partial) = partial() else {
        return Ok(());
    };
    let started = Instant::now();
    let bytes = partial.fetch(in_layers(layers))?;
    if bytes > 0
        && let Some(mapped) = loaded.mapped_layers(layers)
    {
        crate::engine::backend::refresh_weights(mapped);
    }
    if bytes > 0 {
        log::info!(
            "orangu-server: fetched layers {}..{} ({}) in {:.1} s",
            layers.start,
            layers.end,
            orangu::format::format_bytes(bytes),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// Fetches `layers`' tensors again, from the next source: they failed the
/// parent's check. The sources they came from are not asked again.
pub fn refetch(loaded: &LoadedModel, layers: &Range<usize>) -> Result<()> {
    let Some(partial) = partial() else {
        bail!("this node runs no partial model");
    };
    let dropped = partial.forget(in_layers(layers), true)?;
    log::warn!(
        "orangu-server: layers {}..{} as fetched do not match the parent's model; fetching them \
         again{}",
        layers.start,
        layers.end,
        if dropped.is_empty() {
            String::new()
        } else {
            format!(" without {}", dropped.join(", "))
        }
    );
    self::layers(loaded, layers).context("fetching the layers again")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workers::protocol::ProbeState;

    fn found(node: &str, load: f32, features: u64, content: u8, model: u8) -> Found {
        Found {
            addr: format!("127.0.0.1:{}", 9000 + node.len()),
            node: node.to_string(),
            name: format!("{node}:8400"),
            state: ProbeState::Idle,
            load,
            whole: true,
            api: String::new(),
            model: [model; 32],
            content: [content; 32],
            workers: 0,
            label: "bartowski/Model-GGUF".to_string(),
            quant: "Q4_K_M".to_string(),
            features,
            pace: Default::default(),
            demand: Vec::new(),
            kind: Default::default(),
            capabilities: Vec::new(),
            rtt: Duration::from_millis(1),
            alternates: Vec::new(),
        }
    }

    /// Only a node that advertised the bit, holds the whole file and serves
    /// the same model is asked; the parent before the others, then the
    /// least busy.
    #[test]
    fn sources_are_the_parent_then_the_least_busy() {
        let nodes = [
            found("busy", 0.9, FEATURE_FETCH, 2, 1),
            found("idle", 0.1, FEATURE_FETCH, 2, 1),
            found("old", 0.0, 0, 2, 1),
            found("partial", 0.0, FEATURE_FETCH, 0, 1),
            found("other", 0.0, FEATURE_FETCH, 2, 9),
            found("parent", 0.5, FEATURE_FETCH, 2, 1),
        ];
        let names = |parent| -> Vec<String> {
            in_order(&nodes, parent, [1; 32])
                .iter()
                .map(|f| f.node.clone())
                .collect()
        };
        assert_eq!(names(Some("parent")), ["parent", "idle", "busy"]);
        assert_eq!(names(None), ["idle", "parent", "busy"]);
        // A parent that serves no bytes is not asked first, or at all.
        assert_eq!(names(Some("old")), ["idle", "parent", "busy"]);
    }

    /// At start, a node serves the model a spec names only with the bit and
    /// the whole file.
    #[test]
    fn a_spec_is_served_by_a_whole_node_with_the_bit() {
        let spec = "bartowski/Model-GGUF:Q4_K_M";
        assert!(serves(&found("a", 0.0, FEATURE_FETCH, 2, 1), spec));
        assert!(serves(
            &found("a", 0.0, FEATURE_FETCH, 2, 1),
            "bartowski/Model-GGUF"
        ));
        assert!(!serves(&found("a", 0.0, 0, 2, 1), spec));
        assert!(!serves(&found("a", 0.0, FEATURE_FETCH, 0, 1), spec));
        assert!(!serves(
            &found("a", 0.0, FEATURE_FETCH, 2, 1),
            "bartowski/Model-GGUF:Q8_0"
        ));
    }
}
