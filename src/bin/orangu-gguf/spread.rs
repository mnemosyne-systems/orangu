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

//! Quantization divided over the gguf nodes of a channel.
//!
//! Every row of a tensor encodes on its own, so a tensor's rows can be cut
//! into pieces, encoded on different nodes and written in row order: the
//! bytes are the ones a single machine writes. Each node reads the rows
//! from its own copy of the source and sends back only the encoded bytes.
//!
//! Whether a tensor is divided is decided tensor by tensor, per type, from
//! what has been measured: how fast this node encodes the type alone, and
//! how fast the tensors of that type that were divided went, from the
//! first piece sent to the last one back. Until a type has been divided
//! once past its first tensor, the nodes are taken to be as fast as this
//! one, behind their links' measured latency and bandwidth and the time
//! each takes to read the tensor. The decision is logged with its numbers
//! whenever it changes for a type.

use anyhow::Result;
use orangu::gguf_write::TensorPlan;
use orangu::quantize::type_name;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use crate::jobs::{self, Bytes, ModelRef, Piece, Report};
use crate::node::Claimed;

/// The most encoded bytes one piece carries.
const MAX_PIECE_BYTES: usize = 64 << 20;
/// Pieces per node in a divided tensor, so a faster node takes more.
const PIECES_PER_NODE: usize = 4;
/// How much sooner dividing must finish before a tensor is divided.
const MARGIN: f64 = 0.9;
/// What a link's bandwidth is timed with.
const PROBE_BYTES: u64 = 4 << 20;

/// A link's measured round trip and bandwidth towards this node.
struct Link {
    latency: f64,
    bandwidth: f64,
}

/// Elements and the seconds they took, added up.
#[derive(Clone, Copy, Default)]
struct Pace {
    elements: f64,
    seconds: f64,
}

impl Pace {
    fn add(&mut self, elements: f64, seconds: f64) {
        self.elements += elements;
        self.seconds += seconds;
    }
    fn rate(&self) -> Option<f64> {
        (self.seconds > 0.0).then(|| self.elements / self.seconds)
    }
}

/// What has been measured of one type.
#[derive(Default)]
struct Measured {
    /// Encoded here alone.
    here: Pace,
    /// Divided, past the first tensor.
    divided: Pace,
    /// Tensors divided so far.
    tries: usize,
    /// The last decision logged, and whether it was from a measured pace.
    said: Option<(bool, bool)>,
    /// Tensors divided and encoded alone.
    count: (usize, usize),
}

/// The claimed nodes of one conversion and what has been measured of them.
pub struct Spread {
    links: Vec<Claimed>,
    link: HashMap<String, Link>,
    types: BTreeMap<u32, Measured>,
    report: Report,
    /// Divides every tensor it can and slows this node's pieces, so a
    /// test sees every node take some.
    #[cfg(test)]
    forced: bool,
}

impl Spread {
    /// Opens `source` on every claimed node and times each link; `None`
    /// when no node is left to divide with.
    pub fn start(source: &Path, requantize: bool, mut links: Vec<Claimed>) -> Result<Option<Self>> {
        if links.is_empty() {
            return Ok(None);
        }
        let mut open = Bytes::default();
        open.string("quantize");
        ModelRef::of(source)?.encode(&mut open);
        open.u64(requantize as u64);
        jobs::run::<()>(
            0,
            &|_| Ok(()),
            &mut links,
            "quantize",
            Some(&open.0),
            &|_| Vec::new(),
            &|_| Ok(()),
            &mut |_, _| Ok(()),
        )?;
        let mut link = HashMap::new();
        links.retain_mut(|l| match (echo(l, 0), echo(l, PROBE_BYTES)) {
            (Ok(latency), Ok(full)) => {
                let measured = Link {
                    latency,
                    bandwidth: PROBE_BYTES as f64 / (full - latency).max(1e-6),
                };
                eprintln!(
                    "quantize: {}: {:.1} ms a round trip, {}/s",
                    l.found.name,
                    measured.latency * 1e3,
                    crate::bytes(measured.bandwidth as u64)
                );
                link.insert(l.found.name.clone(), measured);
                true
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("quantize: {} left out: {e:#}", l.found.name);
                false
            }
        });
        if links.is_empty() {
            return Ok(None);
        }
        Ok(Some(Spread {
            links,
            link,
            types: BTreeMap::new(),
            report: Report::default(),
            #[cfg(test)]
            forced: false,
        }))
    }

    /// Encodes `values`, the whole of tensor `plan`, into `sink`: divided
    /// over the nodes when that is measured to finish sooner, here
    /// otherwise. `read` is how long reading the tensor took here.
    pub fn tensor(
        &mut self,
        plan: &TensorPlan,
        values: &[f32],
        importance: Option<&[f32]>,
        read: f64,
        sink: &mut (dyn FnMut(&[u8]) -> Result<()> + Send),
    ) -> Result<()> {
        let ty = plan.ggml_type;
        let ncols = plan.ncols().max(1);
        let rows = values.len() / ncols;
        let importance = importance.unwrap_or(&[]);
        let piece = |first: usize, count: usize| Piece {
            name: &plan.name,
            ggml_type: ty,
            first,
            count,
            rows,
            importance,
        };
        let workers = self.links.len() + 1;
        if self.links.is_empty() || rows < PIECES_PER_NODE * workers {
            return self.alone(&piece(0, rows), values, sink);
        }
        let here = match self.types.get(&ty).and_then(|m| m.here.rate()) {
            Some(rate) => rate,
            None => self.sample(&piece(0, rows), values),
        };

        let elements = values.len() as f64;
        let alone = elements / here;
        let measured = self.types.entry(ty).or_default();
        let (divided, basis) = match measured.divided.rate() {
            Some(rate) => (
                elements / rate,
                format!("measured {} divided", per_second(rate)),
            ),
            None => {
                // Every node reads the tensor before its first piece; then
                // it encodes as fast as this one, and its bytes cross the
                // link.
                let per_element = plan.bytes() as f64 / elements;
                let mut rate = here;
                let mut setup = 0.0f64;
                let mut speeds = Vec::new();
                for link in self.link.values() {
                    rate += 1.0 / (1.0 / here + per_element / link.bandwidth);
                    setup = setup.max(read + link.latency);
                    speeds.push(format!("{}/s", crate::bytes(link.bandwidth as u64)));
                }
                (
                    setup + elements / rate,
                    format!(
                        "nodes taken as fast as here, over {} links, reading the tensor first \
                         ({read:.2} s)",
                        speeds.join(", ")
                    ),
                )
            }
        };
        #[allow(unused_mut)]
        let mut divide = divided < alone * MARGIN;
        #[cfg(test)]
        {
            divide |= self.forced;
        }
        let said = (divide, measured.divided.rate().is_some());
        if measured.said.replace(said) != Some(said) {
            eprintln!(
                "\nquantize: {} at {}: alone {alone:.2} s ({} here), divided over {workers} \
                 nodes {divided:.2} s ({basis}) → {}",
                type_name(ty),
                plan.name,
                per_second(here),
                if divide { "dividing" } else { "alone" }
            );
        }
        if !divide {
            return self.alone(&piece(0, rows), values, sink);
        }

        let k = (PIECES_PER_NODE * workers)
            .max(plan.bytes().div_ceil(MAX_PIECE_BYTES))
            .min(rows);
        let at = |i: usize| {
            let first = rows * i / k;
            piece(first, rows * (i + 1) / k - first)
        };
        #[cfg(test)]
        let forced = self.forced;
        let started = Instant::now();
        let report = jobs::run(
            k,
            &|i| {
                #[cfg(test)]
                if forced {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                Ok(at(i).quantize(values))
            },
            &mut self.links,
            "quantize",
            None,
            &|i| at(i).encode(),
            &|data| Ok(data.to_vec()),
            &mut |_, bytes| sink(&bytes),
        )?;
        let seconds = started.elapsed().as_secs_f64();
        let measured = self.types.entry(ty).or_default();
        measured.count.0 += 1;
        // The first divided tensor of a type also builds the nodes' tables.
        if measured.tries > 0 {
            measured.divided.add(elements, seconds);
        }
        measured.tries += 1;
        self.report.absorb(report);
        Ok(())
    }

    /// Encodes the whole tensor here, and keeps its pace.
    fn alone(
        &mut self,
        whole: &Piece<'_>,
        values: &[f32],
        sink: &mut (dyn FnMut(&[u8]) -> Result<()> + Send),
    ) -> Result<()> {
        let started = Instant::now();
        let bytes = whole.quantize(values);
        let measured = self.types.entry(whole.ggml_type).or_default();
        measured
            .here
            .add(values.len() as f64, started.elapsed().as_secs_f64());
        measured.count.1 += 1;
        sink(&bytes)
    }

    /// This node's pace for a type not encoded yet, timed on a few rows.
    fn sample(&mut self, whole: &Piece<'_>, values: &[f32]) -> f64 {
        let rows = (rayon::current_num_threads() * 4).min(whole.rows);
        let ncols = values.len() / whole.rows;
        // One row first: a type's tables are built on its first use.
        Piece { count: 1, ..*whole }.quantize(values);
        let started = Instant::now();
        Piece {
            count: rows,
            ..*whole
        }
        .quantize(values);
        let seconds = started.elapsed().as_secs_f64().max(1e-9);
        let measured = self.types.entry(whole.ggml_type).or_default();
        measured.here.add((rows * ncols) as f64, seconds);
        measured.here.rate().unwrap_or(f64::MAX)
    }

    /// Logs what was divided and who encoded it.
    pub fn finish(self) {
        eprintln!();
        for (&ty, m) in &self.types {
            eprintln!(
                "quantize: {}: {} tensors divided, {} encoded alone",
                type_name(ty),
                m.count.0,
                m.count.1
            );
        }
        self.report.print("quantize");
    }
}

/// The time one echo of `bytes` takes on `link`.
fn echo(link: &mut Claimed, bytes: u64) -> Result<f64> {
    let mut w = Bytes::default();
    w.u64(bytes);
    let started = Instant::now();
    let back = jobs::call(link, "echo", w.0)?;
    anyhow::ensure!(back.len() as u64 == bytes, "the echo came back short");
    Ok(started.elapsed().as_secs_f64())
}

fn per_second(elements: f64) -> String {
    format!("{:.1} M elements/s", elements / 1e6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Settings, Worker, claim};
    use orangu::gguf::GgufValue;
    use orangu::quantize::{self as q, GGML_TYPE_F32, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_XS};

    fn settings() -> Settings {
        Settings {
            name: "spread".to_string(),
            secret: Some("s".to_string()),
            host: "127.0.0.1".to_string(),
            port: 0,
            server_port: 0,
            networks: Vec::new(),
            peers: Vec::new(),
            scan: false,
            tls: None,
        }
    }

    fn values(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..rows * cols)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn plan(name: &str, cols: usize, rows: usize, ggml_type: u32) -> TensorPlan {
        TensorPlan {
            name: name.to_string(),
            dims: vec![cols as u64, rows as u64],
            ggml_type,
        }
    }

    /// The same bytes, tensor by tensor, on one, two and three nodes; a
    /// stacked tensor's experts keep their own importances across pieces.
    #[test]
    fn a_quantized_tensor_is_the_same_on_one_two_and_three_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.gguf");
        let shapes = [("a.weight", 512, 48, 1), ("b.weight", 256, 40, 2)];
        let sources: Vec<TensorPlan> = shapes
            .iter()
            .map(|&(name, cols, rows, _)| plan(name, cols, rows, GGML_TYPE_F32))
            .collect();
        orangu::gguf_write::write_weighted(
            &path,
            &[(
                "general.architecture".to_string(),
                GgufValue::String("llama".to_string()),
            )],
            &sources,
            |p| {
                Ok(values(
                    p.elements() / p.ncols(),
                    p.ncols(),
                    p.ncols() as u64,
                ))
            },
            &|_| None,
            &|_, _, _| {},
        )
        .unwrap();

        // Each tensor as each type, with and without an importance.
        let mut cases = Vec::new();
        for &(name, cols, rows, experts) in &shapes {
            let importance: Vec<f32> = (0..cols * experts).map(|i| 0.5 + (i % 7) as f32).collect();
            for ty in [GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_XS, q::GGML_TYPE_Q4_K] {
                cases.push((plan(name, cols, rows, ty), Some(importance.clone())));
                if ty != GGML_TYPE_IQ2_XS {
                    cases.push((plan(name, cols, rows, ty), None));
                }
            }
        }
        let expected: Vec<Vec<u8>> = cases
            .iter()
            .map(|(p, importance)| {
                let v = values(p.elements() / p.ncols(), p.ncols(), p.ncols() as u64);
                match importance {
                    Some(w) => q::encode_weighted(p.ggml_type, &v, p.ncols(), w),
                    None => q::encode(p.ggml_type, &v, p.ncols()),
                }
            })
            .collect();

        let workers: Vec<_> = (0..2)
            .map(|_| Worker::start(settings()).unwrap().1)
            .collect();
        let hash = orangu::workers::channel::channel_hash("spread", Some("s"));
        for nodes in [2, 3] {
            let found: Vec<_> = workers[..nodes - 1]
                .iter()
                .map(|addr| orangu::workers::channel::probe(*addr, &hash, None).unwrap())
                .collect();
            let links = claim(&settings(), &format!("top{nodes}"), &found, "quantize");
            assert_eq!(links.len(), nodes - 1);
            let mut spread = Spread::start(&path, false, links).unwrap().unwrap();
            spread.forced = true;
            for ((p, importance), expected) in cases.iter().zip(&expected) {
                let v = values(p.elements() / p.ncols(), p.ncols(), p.ncols() as u64);
                let mut out = Vec::new();
                spread
                    .tensor(p, &v, importance.as_deref(), 0.0, &mut |b| {
                        out.extend_from_slice(b);
                        Ok(())
                    })
                    .unwrap();
                assert!(
                    out == *expected,
                    "{} as {} on {nodes} nodes",
                    p.name,
                    q::type_name(p.ggml_type)
                );
            }
            let report = &spread.report;
            assert!(report.lost.is_empty(), "{report:?}");
            assert_eq!(report.by_node.len(), nodes, "{report:?}");
        }
    }

    /// A node refuses a piece that does not fit its tensor, rather than
    /// failing inside the encoder.
    #[test]
    fn a_piece_that_does_not_fit_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.gguf");
        orangu::gguf_write::write_weighted(
            &path,
            &[],
            &[plan("a.weight", 256, 8, GGML_TYPE_F32)],
            |_| Ok(values(8, 256, 1)),
            &|_| None,
            &|_, _, _| {},
        )
        .unwrap();
        let mut open = Bytes::default();
        open.string("quantize");
        ModelRef::of(&path).unwrap().encode(&mut open);
        open.u64(0);
        let mut session = jobs::Session::default();
        jobs::serve(&mut session, "open", &open.0).unwrap();
        let ask = |session: &mut jobs::Session, first, count, importance: &[f32]| {
            let piece = Piece {
                name: "a.weight",
                ggml_type: GGML_TYPE_IQ4_XS,
                first,
                count,
                rows: 8,
                importance,
            };
            jobs::serve(session, "quantize", &piece.encode())
        };
        assert!(ask(&mut session, 2, 3, &[]).is_ok());
        assert!(ask(&mut session, 6, 3, &[]).is_err());
        assert!(ask(&mut session, 0, 8, &[1.0; 100]).is_err());
        assert!(ask(&mut session, 0, 8, &[1.0; 256 * 3]).is_err());
    }
}
