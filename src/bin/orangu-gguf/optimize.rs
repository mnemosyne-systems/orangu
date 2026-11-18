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

//! `-q auto`: a type per tensor, chosen for this machine's speed within
//! the KL and size of the standard `Q4_K_M`.
//!
//! The standard mixture is planned first; it is the reference. Every tensor
//! it quantizes gets an error table (`auto`), every shape a speed table on
//! the device the model will decode on (`speed`), and the choice (`choose`)
//! replaces the reference's types. Tensors the reference keeps as they are
//! stay that way.
//!
//! What a unit of a tensor's error costs in KL depends on the tensor's role
//! and on the model — the output head's lands on the logits directly, and
//! one architecture's value projection matters more than another's — so it
//! is measured on the model itself: the standard mixture is encoded in
//! memory and read against the source over a text (`kl`), then each role in
//! turn is lowered, and the KL it adds per unit of error it adds is that
//! role's weight. The choice is then measured the same way; one over the
//! standard's KL is chosen again under a tighter budget, a few times at
//! most, and the standard mixture is written when none comes within it.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use orangu::engine::loader::{LoadedModel, TensorBytes};
use orangu::gguf::{GgufFile, GgufValue};
use orangu::quantize as q;

use crate::auto::{self, Cost};
use crate::choose::{self, Tensor};
use crate::kl;
use crate::quant;
use crate::speed::{self, PREFILL_TOKENS, Shape};
use crate::write::TensorPlan;

/// What the chosen file may take beyond the reference's limits.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The most bytes of weights; the reference's own when `None`.
    pub max_bytes: Option<u64>,
    /// The most KL from the source over the calibration text, in nats; the
    /// standard mixture's own when `None`.
    pub max_kl: Option<f64>,
    /// The text each choice's KL is measured over.
    pub text: Option<std::path::PathBuf>,
    /// Types pinned from the command line, which the choice keeps.
    pub overrides: Vec<Override>,
}

/// Room a card keeps beside the weights for KV pages and prompt scratch.
/// A file that leaves it less runs prompts with the KV on the host.
const DEVICE_HEADROOM: u64 = 3 << 29;

/// What the choice did, for the summary.
pub struct Summary {
    pub reference: choose::Choice,
    pub chosen: choose::Choice,
    pub device: String,
    /// The KL measured for the standard mixture and for the choice.
    pub standard_kl: f64,
    pub chosen_kl: f64,
    /// Whether no choice came within the KL, so the standard mixture is
    /// what is written.
    pub kept_standard: bool,
}

fn expert_used(metadata: &[(String, GgufValue)]) -> usize {
    let arch = metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| match v {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("");
    let key = format!("{arch}.expert_used_count");
    metadata
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| match v {
            GgufValue::U32(n) => Some(*n as usize),
            GgufValue::I32(n) => Some(*n as usize),
            GgufValue::U64(n) => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(1)
}

/// The role a tensor plays: its name without the block number, so every
/// block's value projection is one role.
fn role(name: &str) -> String {
    let base = name.trim_end_matches(".weight");
    match base.split_once('.') {
        Some(("blk", rest)) => rest.split_once('.').map_or(rest, |(_, r)| r).to_string(),
        _ => base.to_string(),
    }
}

/// The type a role is lowered to when its weight is measured: the largest
/// candidate at most about three quarters of the reference type's bytes,
/// leaving out the codebook types, whose errors fall differently; `None`
/// when nothing is smaller.
fn lowered(costs: &[Cost], reference: u32) -> Option<usize> {
    let at = costs.iter().find(|c| c.ggml_type == reference)?.bytes;
    let usable = |c: &&Cost| c.bytes < at && c.ggml_type != q::GGML_TYPE_IQ4_XS;
    let within = costs
        .iter()
        .filter(usable)
        .filter(|c| c.bytes * 4 <= at * 3 + at / 8)
        .max_by_key(|c| c.bytes);
    let pick = within.or_else(|| costs.iter().filter(usable).min_by_key(|c| c.bytes))?;
    costs.iter().position(|c| c.ggml_type == pick.ggml_type)
}

/// How many of a tensor's products one token runs: none for a table read
/// one row at a time, the routed count for a stack of experts, else one.
fn runs_per_token(plan: &TensorPlan, tied: bool, experts_used: usize) -> f64 {
    match plan.name.as_str() {
        "token_embd.weight" if !tied => 0.0,
        "per_layer_token_embd.weight" => 0.0,
        _ if plan.dims.len() >= 3 => experts_used as f64,
        _ => 1.0,
    }
}

/// Whether the choice may change this tensor: one the reference quantizes.
fn adjustable(plan: &TensorPlan) -> bool {
    plan.dims.len() >= 2
        && !matches!(
            plan.ggml_type,
            q::GGML_TYPE_F32 | q::GGML_TYPE_F16 | q::GGML_TYPE_BF16
        )
}

/// The `general.file_type` of a file whose weights are `plans`: the type
/// most of its quantized bytes are in, as the standard name for it.
pub fn file_type(plans: &[TensorPlan]) -> u32 {
    let mut by_type: BTreeMap<u32, u64> = BTreeMap::new();
    for p in plans.iter().filter(|p| adjustable(p)) {
        let elements: u64 = p.dims.iter().product();
        let ncols = p.dims[0] as usize;
        *by_type.entry(p.ggml_type).or_default() +=
            (q::row_bytes(p.ggml_type, ncols) as u64) * (elements / ncols.max(1) as u64);
    }
    let main = by_type
        .iter()
        .max_by_key(|&(t, b)| (*b, std::cmp::Reverse(*t)))
        .map_or(q::GGML_TYPE_Q4_K, |(t, _)| *t);
    use quant::Ftype;
    match main {
        q::GGML_TYPE_Q2_K => Ftype::Q2K.file_type(),
        q::GGML_TYPE_Q3_K => Ftype::Q3KM.file_type(),
        q::GGML_TYPE_Q5_K => Ftype::Q5KM.file_type(),
        q::GGML_TYPE_Q6_K => Ftype::Q6K.file_type(),
        q::GGML_TYPE_IQ4_XS => Ftype::IQ4XS.file_type(),
        // The ggml file type of a mostly-`Q8_0` file.
        q::GGML_TYPE_Q8_0 => 7,
        _ => Ftype::Q4KM.file_type(),
    }
}

/// Replaces the types of `plans` (the reference mixture, in file order)
/// with the choice for this machine. `values` reads a tensor's weights from
/// `source`, and `text` is what the KL is measured over.
#[allow(clippy::too_many_arguments)]
pub fn select(
    file: &GgufFile,
    source: &Path,
    text: &kl::Text,
    plans: &mut [TensorPlan],
    importance: impl Fn(&str) -> Option<Vec<f32>>,
    values: impl Fn(&TensorPlan) -> Result<Vec<f32>>,
    options: &Options,
) -> Result<Summary> {
    let tied = !plans.iter().any(|p| p.name == "output.weight");
    let experts_used = expert_used(&file.metadata);
    let picked: Vec<usize> = (0..plans.len())
        .filter(|&i| adjustable(&plans[i]))
        .collect();
    if picked.is_empty() {
        bail!("nothing in this file is quantized, so there is nothing to choose");
    }
    let reference_types: Vec<u32> = picked.iter().map(|&i| plans[i].ggml_type).collect();

    // The error table: every candidate, the reference's own type and a
    // pinned one among them.
    let started = std::time::Instant::now();
    let mut tables: Vec<Vec<Cost>> = Vec::with_capacity(picked.len());
    for (n, &i) in picked.iter().enumerate() {
        let plan = &plans[i];
        print!("\r  measuring {}/{} {}    ", n + 1, picked.len(), plan.name);
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let weights = values(plan).with_context(|| plan.name.clone())?;
        let ncols = plan.dims[0] as usize;
        let matrix = importance(&plan.name);
        let pin = pinned(&options.overrides, &plan.name);
        let mut costs = auto::costs(&weights, ncols, matrix.as_deref());
        let rows = weights.len() / ncols.max(1);
        for wanted in [Some(plan.ggml_type), pin].into_iter().flatten() {
            if costs.iter().any(|c| c.ggml_type == wanted) {
                continue;
            }
            costs.push(Cost {
                ggml_type: wanted,
                bytes: (q::row_bytes(wanted, ncols) * rows) as u64,
                error: auto::estimated_error(
                    wanted,
                    &weights,
                    ncols,
                    matrix.as_deref(),
                    auto::SAMPLE_ROWS,
                ),
            });
        }
        tables.push(costs);
    }
    println!(
        "\r  measured {} tensors' errors in {:.1} s{:30}",
        picked.len(),
        started.elapsed().as_secs_f64(),
        ""
    );

    // The speed table, on the device the model decodes on — and, for a
    // model whose standard file overflows that card, its routed experts on
    // the cores, where the server keeps them.
    let started = std::time::Instant::now();
    let devices = speed::devices();
    let (device, cores) = (&devices[0], devices.last().expect("the cores are a device"));
    let reference_bytes: u64 = (0..picked.len())
        .map(|n| {
            tables[n]
                .iter()
                .find(|c| c.ggml_type == reference_types[n])
                .map_or(0, |c| c.bytes)
        })
        .sum();
    let overflows = match device {
        speed::Device::Gpu {
            memory: Some(total),
            ..
        } => reference_bytes > total.saturating_sub(DEVICE_HEADROOM),
        _ => false,
    };
    let on_host = |i: usize| overflows && plans[i].dims.len() >= 3;
    let mut rates: BTreeMap<(bool, u32, Shape), Option<speed::Rate>> = BTreeMap::new();
    for (&i, costs) in picked.iter().zip(&tables) {
        if runs_per_token(&plans[i], tied, experts_used) == 0.0 {
            continue;
        }
        let shape = Shape {
            in_dim: plans[i].dims[0] as usize,
            out_dim: plans[i].dims[1] as usize,
        };
        let host = on_host(i);
        for c in costs {
            rates.entry((host, c.ggml_type, shape)).or_insert_with(|| {
                speed::rate(if host { cores } else { device }, c.ggml_type, shape)
            });
        }
    }
    println!(
        "  timed {} type × shape pairs on {}{} in {:.1} s",
        rates.len(),
        device.name(),
        if overflows {
            format!(" and the routed experts on {}", cores.name())
        } else {
            String::new()
        },
        started.elapsed().as_secs_f64()
    );
    let mut times: Vec<Vec<Option<[[f64; 2]; 2]>>> = Vec::with_capacity(picked.len());
    for (&i, costs) in picked.iter().zip(&tables) {
        let plan = &plans[i];
        let runs = runs_per_token(plan, tied, experts_used);
        let shape = Shape {
            in_dim: plan.dims[0] as usize,
            out_dim: plan.dims[1] as usize,
        };
        let row: Vec<Option<[[f64; 2]; 2]>> = costs
            .iter()
            .map(|c| {
                if runs == 0.0 {
                    return Some([[0.0; 2]; 2]);
                }
                rates
                    .get(&(on_host(i), c.ggml_type, shape))
                    .cloned()
                    .flatten()
                    .map(|r| {
                        [
                            r.decode_us.map(|us| us * runs),
                            r.prefill_us.map(|us| us * runs / PREFILL_TOKENS as f64),
                        ]
                    })
            })
            .collect();
        let own = costs.iter().position(|c| c.ggml_type == plan.ggml_type);
        if own.is_none_or(|c| row[c].is_none()) {
            bail!(
                "{} could not time {} at {}×{}",
                device.name(),
                q::type_name(plan.ggml_type),
                shape.in_dim,
                shape.out_dim
            );
        }
        times.push(row);
    }

    // The model in memory, at the reference's types, read against the
    // source.
    let started = std::time::Instant::now();
    let mut model = LoadedModel::open(source)
        .with_context(|| format!("loading {} to measure", source.display()))?;
    let reference_logits = kl::reference(&model, text)?;
    let encode = |n: usize, ggml_type: u32| -> Result<TensorBytes> {
        let plan = &plans[picked[n]];
        let weights = values(plan)?;
        let ncols = plan.dims[0] as usize;
        let bytes = match importance(&plan.name).filter(|_| q::takes_importance(ggml_type)) {
            Some(w) => q::encode_weighted(ggml_type, &weights, ncols, &w),
            None => q::encode(ggml_type, &weights, ncols),
        };
        Ok(Arc::new(bytes))
    };
    let mut standard: Vec<TensorBytes> = Vec::with_capacity(picked.len());
    for n in 0..picked.len() {
        let bytes = encode(n, reference_types[n])?;
        model.replace_tensor(&plans[picked[n]].name, reference_types[n], bytes.clone())?;
        standard.push(bytes);
    }
    let standard_kl = kl::kl(&model, text, &reference_logits)?;
    println!(
        "  the standard Q4_K_M reads at KL {standard_kl:.4} over {} tokens ({:.1} s)",
        text.positions(),
        started.elapsed().as_secs_f64()
    );

    // Each role's weight: the KL it adds lowered, per unit of the error the
    // table says lowering it adds.
    let started = std::time::Instant::now();
    let mut members: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (n, &i) in picked.iter().enumerate() {
        members.entry(role(&plans[i].name)).or_default().push(n);
    }
    let total_bytes: u64 = (0..picked.len())
        .map(|n| {
            tables[n]
                .iter()
                .find(|c| c.ggml_type == reference_types[n])
                .map_or(0, |c| c.bytes)
        })
        .sum();
    let mut measured: BTreeMap<String, f64> = BTreeMap::new();
    for (role, ns) in &members {
        let lower: Vec<(usize, usize)> = ns
            .iter()
            .filter_map(|&n| lowered(&tables[n], reference_types[n]).map(|c| (n, c)))
            .collect();
        let bytes: u64 = ns
            .iter()
            .map(|&n| {
                tables[n]
                    .iter()
                    .find(|c| c.ggml_type == reference_types[n])
                    .map_or(0, |c| c.bytes)
            })
            .sum();
        if lower.is_empty() || bytes * 200 < total_bytes {
            continue;
        }
        let mut added_error = 0.0;
        for &(n, c) in &lower {
            let own = tables[n]
                .iter()
                .find(|x| x.ggml_type == reference_types[n])
                .map_or(0.0, |x| x.error);
            added_error += tables[n][c].error - own;
            let t = tables[n][c].ggml_type;
            model.replace_tensor(&plans[picked[n]].name, t, encode(n, t)?)?;
        }
        let lowered_kl = kl::kl(&model, text, &reference_logits)?;
        for &(n, _) in &lower {
            model.replace_tensor(
                &plans[picked[n]].name,
                reference_types[n],
                standard[n].clone(),
            )?;
        }
        if added_error > 0.0 {
            measured.insert(role.clone(), (lowered_kl - standard_kl) / added_error);
        }
    }
    let mut positive: Vec<f64> = measured.values().copied().filter(|w| *w > 0.0).collect();
    positive.sort_by(f64::total_cmp);
    let median = positive.get(positive.len() / 2).copied().unwrap_or(1.0);
    let floor = median * 0.1;
    let weight_of = |role: &str| measured.get(role).map_or(median, |w| w.max(floor));
    println!(
        "  measured {} roles' weights in {:.1} s",
        measured.len(),
        started.elapsed().as_secs_f64()
    );

    // A card chooses its prompt kernels apart from its decode kernels, both
    // for the whole file: prompts are priced on the family faster for the
    // reference's, and the choice's decode family is left to `choose`.
    let prompt_family = (0..2)
        .min_by(|&a, &b| {
            let total = |f: usize| -> f64 {
                (0..picked.len())
                    .filter_map(|n| {
                        let c = tables[n]
                            .iter()
                            .position(|c| c.ggml_type == reference_types[n])?;
                        times[n][c].map(|t| t[1][f])
                    })
                    .sum()
            };
            total(a).total_cmp(&total(b))
        })
        .unwrap_or(1);

    // The choice, under the weighted error.
    let small_wanted = options.max_bytes.is_some_and(|max| max < total_bytes);
    let mut tensors: Vec<Tensor> = Vec::with_capacity(picked.len());
    let mut all: Vec<Tensor> = Vec::with_capacity(picked.len());
    for n in 0..picked.len() {
        let plan = &plans[picked[n]];
        let w = weight_of(&role(&plan.name));
        let pin = pinned(&options.overrides, &plan.name);
        let mut full = Tensor {
            costs: Vec::new(),
            time_us: Vec::new(),
            on_device: !on_host(picked[n]),
        };
        let mut kept = Tensor {
            costs: Vec::new(),
            time_us: Vec::new(),
            on_device: !on_host(picked[n]),
        };
        for (c, cost) in tables[n].iter().enumerate() {
            let Some(mut time) = times[n][c] else {
                continue;
            };
            time[1] = [time[1][prompt_family]; 2];
            let cost = Cost {
                error: cost.error * w,
                ..cost.clone()
            };
            let t = cost.ggml_type;
            let allowed = match pin {
                Some(p) => t == p,
                None => small_wanted || t == reference_types[n] || !auto::SMALL.contains(&t),
            };
            if allowed {
                kept.costs.push(cost.clone());
                kept.time_us.push(time);
            }
            full.costs.push(cost);
            full.time_us.push(time);
        }
        tensors.push(kept);
        all.push(full);
    }
    let reference = choose::reference(&all, &reference_types)
        .context("the reference's own types are among the candidates")?;
    let device_room = match device {
        speed::Device::Gpu { memory, .. } => memory.map(|total| {
            total
                .saturating_sub(DEVICE_HEADROOM)
                .max(reference.device_bytes)
        }),
        speed::Device::Cpu { .. } => None,
    };
    let mut limits = choose::default_limits(&reference, device_room);
    if let Some(max) = options.max_bytes {
        limits.max_bytes = max;
    }
    let target = options.max_kl.unwrap_or(standard_kl);
    limits.max_error = reference.error * target / standard_kl.max(1e-12);
    let prices = (
        reference.decode_us,
        reference.prefill_us,
        reference.bytes as f64,
    );

    // Every block's tensor of one role takes one type. A role whose blocks
    // differ can decode slower than either of its types would alone, which
    // a speed table of single products cannot see.
    let groups = role_groups(&picked, plans, &tensors, &options.overrides);
    let grouped: Vec<Tensor> = groups.iter().map(|ns| merged(&tensors, ns)).collect();

    // Measured, and the budget tightened by what the measurement says, at
    // most [`ROUNDS`] times. A choice no round brings within the KL is not
    // written: the standard mixture is.
    let mut outcome = None;
    for round in 1..=ROUNDS {
        let by_group = choose::choose(&grouped, &limits, prices);
        let mut types = reference_types.clone();
        for (ns, &t) in groups.iter().zip(&by_group.types) {
            for &n in ns {
                types[n] = t;
            }
        }
        let chosen = choose::Choice {
            within: by_group.within,
            ..choose::evaluate(&all, &types, by_group.family)
                .context("every chosen type is among its tensor's candidates")?
        };
        for (n, &t) in chosen.types.iter().enumerate() {
            let bytes = if t == reference_types[n] {
                standard[n].clone()
            } else {
                encode(n, t)?
            };
            model.replace_tensor(&plans[picked[n]].name, t, bytes)?;
        }
        let chosen_kl = kl::kl(&model, text, &reference_logits)?;
        println!(
            "  round {round}: predicted KL {:.4}, measured {chosen_kl:.4} against {target:.4}",
            chosen.error / reference.error * standard_kl
        );
        if chosen_kl <= target {
            outcome = Some((chosen, chosen_kl));
            break;
        }
        // By the square of the miss: the measured KL moves less than the
        // prediction does.
        limits.max_error *= (target / chosen_kl).powi(2);
    }
    let kept_standard = outcome.is_none();
    let (chosen, chosen_kl) = match outcome {
        Some(found) => found,
        None => (reference.clone(), standard_kl),
    };
    for (&i, &ggml_type) in picked.iter().zip(&chosen.types) {
        plans[i].ggml_type = ggml_type;
    }
    Ok(Summary {
        reference,
        chosen,
        device: device.name().to_string(),
        standard_kl,
        chosen_kl,
        kept_standard,
    })
}

/// The tensors (positions in `picked`) that take one type together: every
/// block's tensor of a role, a pinned tensor with the others pinned alike,
/// in the order their first member comes. A role whose tensors share no
/// candidate type is left a tensor at a time.
fn role_groups(
    picked: &[usize],
    plans: &[TensorPlan],
    tensors: &[Tensor],
    overrides: &[Override],
) -> Vec<Vec<usize>> {
    let mut index: BTreeMap<(String, Option<u32>), usize> = BTreeMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (n, &i) in picked.iter().enumerate() {
        let name = &plans[i].name;
        let key = (role(name), pinned(overrides, name));
        let g = *index.entry(key).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(n);
    }
    groups
        .into_iter()
        .flat_map(|ns| {
            if merged(tensors, &ns).costs.is_empty() {
                ns.into_iter().map(|n| vec![n]).collect()
            } else {
                vec![ns]
            }
        })
        .collect()
}

/// `members` of `tensors` as one: each type every member can take, at the
/// sum of their errors, bytes and times.
fn merged(tensors: &[Tensor], members: &[usize]) -> Tensor {
    let mut out = Tensor {
        costs: Vec::new(),
        time_us: Vec::new(),
        on_device: members.iter().any(|&n| tensors[n].on_device),
    };
    let Some(&first) = members.first() else {
        return out;
    };
    'types: for cost in &tensors[first].costs {
        let mut sum = Cost {
            ggml_type: cost.ggml_type,
            bytes: 0,
            error: 0.0,
        };
        let mut time = [[0.0; 2]; 2];
        for &n in members {
            let Some(c) = tensors[n]
                .costs
                .iter()
                .position(|x| x.ggml_type == cost.ggml_type)
            else {
                continue 'types;
            };
            sum.bytes += tensors[n].costs[c].bytes;
            sum.error += tensors[n].costs[c].error;
            for (a, b) in time.iter_mut().zip(tensors[n].time_us[c]) {
                a[0] += b[0];
                a[1] += b[1];
            }
        }
        out.costs.push(sum);
        out.time_us.push(time);
    }
    out
}

/// How many times a choice is measured and its budget tightened.
const ROUNDS: usize = 4;

/// The chosen types as a table: how many tensors and bytes each type took,
/// and the predicted cost against the reference.
pub fn print(summary: &Summary, plans: &[TensorPlan]) {
    let mut by_type: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
    for p in plans.iter().filter(|p| adjustable(p)) {
        let elements: u64 = p.dims.iter().product();
        let ncols = p.dims[0] as usize;
        let entry = by_type.entry(p.ggml_type).or_default();
        entry.0 += 1;
        entry.1 += (q::row_bytes(p.ggml_type, ncols) as u64) * (elements / ncols.max(1) as u64);
    }
    println!("\nchosen for {}", summary.device);
    for (t, (n, bytes)) in &by_type {
        println!(
            "  {:>8}  {:>4} tensors  {:>10.1} MiB",
            q::type_name(*t),
            n,
            *bytes as f64 / (1 << 20) as f64
        );
    }
    let (r, c) = (&summary.reference, &summary.chosen);
    let pct = |a: f64, b: f64| (a / b - 1.0) * 100.0;
    if summary.kept_standard {
        println!(
            "  no choice came within KL {:.4}; the standard Q4_K_M is written\n",
            summary.standard_kl
        );
        return;
    }
    println!(
        "  against the standard Q4_K_M: decode {:+.1}% and prompts {:+.1}% predicted, size {:+.1}%, KL {:.4} -> {:.4} measured{}\n",
        pct(c.decode_us, r.decode_us),
        pct(c.prefill_us, r.prefill_us),
        pct(c.bytes as f64, r.bytes as f64),
        summary.standard_kl,
        summary.chosen_kl,
        if c.within {
            ""
        } else {
            " — no choice met every limit; this is the closest"
        }
    );
}

/// A size such as `2.4G`, `2400M`, `512K` or a plain byte count; the units
/// are powers of 1024, and a trailing `B` or `iB` is allowed.
pub fn parse_size(text: &str) -> Result<u64> {
    let t = text.trim();
    let upper = t.to_ascii_uppercase();
    let stripped = upper
        .strip_suffix("IB")
        .or_else(|| upper.strip_suffix('B'))
        .unwrap_or(&upper);
    let (number, scale) = match stripped.chars().last() {
        Some('K') => (&stripped[..stripped.len() - 1], 1u64 << 10),
        Some('M') => (&stripped[..stripped.len() - 1], 1 << 20),
        Some('G') => (&stripped[..stripped.len() - 1], 1 << 30),
        Some('T') => (&stripped[..stripped.len() - 1], 1 << 40),
        _ => (stripped, 1),
    };
    let value: f64 = number
        .trim()
        .parse()
        .with_context(|| format!("{text} is not a size (such as 2.4G or 2400M)"))?;
    if !(value.is_finite() && value > 0.0) {
        bail!("{text} is not a size (such as 2.4G or 2400M)");
    }
    Ok((value * scale as f64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_in_powers_of_1024() {
        assert_eq!(parse_size("2G").unwrap(), 2 << 30);
        assert_eq!(parse_size("1.5GiB").unwrap(), 3 << 29);
        assert_eq!(parse_size("2400m").unwrap(), 2400 << 20);
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert!(parse_size("big").is_err() && parse_size("-1G").is_err());
    }

    fn plan(name: &str, dims: &[u64], ggml_type: u32) -> TensorPlan {
        TensorPlan {
            name: name.into(),
            dims: dims.to_vec(),
            ggml_type,
        }
    }

    #[test]
    fn the_file_type_names_the_type_most_bytes_are_in() {
        let plans = [
            plan("blk.0.ffn_up.weight", &[256, 1024], q::GGML_TYPE_Q5_K),
            plan("blk.0.attn_q.weight", &[256, 256], q::GGML_TYPE_Q6_K),
            plan("blk.0.attn_norm.weight", &[256], q::GGML_TYPE_F32),
        ];
        assert_eq!(file_type(&plans), quant::Ftype::Q5KM.file_type());
    }

    #[test]
    fn a_looked_up_table_runs_no_products() {
        let embd = plan("token_embd.weight", &[256, 1000], q::GGML_TYPE_Q6_K);
        assert_eq!(runs_per_token(&embd, false, 8), 0.0);
        assert_eq!(runs_per_token(&embd, true, 8), 1.0);
        let experts = plan(
            "blk.0.ffn_up_exps.weight",
            &[256, 512, 64],
            q::GGML_TYPE_Q4_K,
        );
        assert_eq!(runs_per_token(&experts, false, 8), 8.0);
    }
}

/// A tensor type pinned from the command line: every tensor whose name the
/// pattern matches is written as the type, standard mixture or choice
/// alike.
#[derive(Clone, Debug)]
pub struct Override {
    pub pattern: regex::Regex,
    pub ggml_type: u32,
}

/// The `ggml_type` named `name` (`q4_k`, `Q6_K`, `bf16`, …).
pub fn type_named(name: &str) -> Option<u32> {
    (0..64u32).find(|&t| {
        let n = q::type_name(t);
        !n.is_empty() && n.eq_ignore_ascii_case(name.trim())
    })
}

/// `PATTERN=TYPE`: a regular expression over tensor names and a type.
pub fn parse_override(text: &str) -> Result<Override> {
    let (pattern, name) = text
        .rsplit_once('=')
        .with_context(|| format!("{text}: expected NAME=TYPE, such as attn_v=q6_k"))?;
    let ggml_type = type_named(name).with_context(|| format!("{name} is not a tensor type"))?;
    let pattern = regex::Regex::new(pattern)
        .with_context(|| format!("{pattern} is not a regular expression"))?;
    Ok(Override { pattern, ggml_type })
}

/// The type `overrides` pin `name` to: the last that matches.
pub fn pinned(overrides: &[Override], name: &str) -> Option<u32> {
    overrides
        .iter()
        .rev()
        .find(|o| o.pattern.is_match(name))
        .map(|o| o.ggml_type)
}

/// Applies `overrides` to the quantized tensors of `plans`. A type whose
/// block does not divide a matched tensor's rows is an error.
pub fn apply_overrides(plans: &mut [TensorPlan], overrides: &[Override]) -> Result<()> {
    for plan in plans.iter_mut().filter(|p| adjustable(p)) {
        if let Some(t) = pinned(overrides, &plan.name) {
            if !(plan.dims[0] as usize).is_multiple_of(q::block_size(t)) {
                bail!(
                    "{} cannot be {}: its rows are {} long",
                    plan.name,
                    q::type_name(t),
                    plan.dims[0]
                );
            }
            plan.ggml_type = t;
        }
    }
    Ok(())
}

#[cfg(test)]
mod override_tests {
    use super::*;

    #[test]
    fn a_pattern_pins_the_tensors_it_matches() {
        let mut plans = vec![
            TensorPlan {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![256, 64],
                ggml_type: q::GGML_TYPE_Q4_K,
            },
            TensorPlan {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![256, 256],
                ggml_type: q::GGML_TYPE_Q4_K,
            },
            TensorPlan {
                name: "blk.0.attn_norm.weight".into(),
                dims: vec![256],
                ggml_type: q::GGML_TYPE_F32,
            },
        ];
        let overrides = [
            parse_override("attn_v=q6_k").unwrap(),
            parse_override("norm=q8_0").unwrap(),
        ];
        apply_overrides(&mut plans, &overrides).unwrap();
        let types: Vec<u32> = plans.iter().map(|p| p.ggml_type).collect();
        assert_eq!(
            types,
            [q::GGML_TYPE_Q6_K, q::GGML_TYPE_Q4_K, q::GGML_TYPE_F32]
        );
        assert!(parse_override("attn_v").is_err() && parse_override("attn_v=q9").is_err());
        assert!(apply_overrides(&mut plans[..1], &[parse_override("v=q4_k").unwrap()]).is_ok());
        let mut narrow = vec![TensorPlan {
            name: "x.weight".into(),
            dims: vec![96, 8],
            ggml_type: q::GGML_TYPE_Q8_0,
        }];
        assert!(apply_overrides(&mut narrow, &[parse_override("x=q4_k").unwrap()]).is_err());
    }
}

#[cfg(test)]
mod role_tests {
    use super::*;

    /// For a real model and importance matrix (`ORANGU_GGUF`,
    /// `ORANGU_IMATRIX`): every role's summed error at the standard
    /// `Q4_K_M`'s types and at each K-quant — the predicted side of a
    /// role's measured KL.
    #[test]
    #[ignore = "needs ORANGU_GGUF and ORANGU_IMATRIX; measures every tensor"]
    fn role_errors_of_a_real_model() {
        let path = std::path::PathBuf::from(std::env::var("ORANGU_GGUF").unwrap());
        let matrix = crate::imatrix::Loaded::open(std::path::Path::new(
            &std::env::var("ORANGU_IMATRIX").unwrap(),
        ))
        .unwrap();
        let file = GgufFile::open(&path).unwrap();
        let model = quant::Model {
            imatrix: true,
            ..quant::Model::from_metadata(&file.metadata)
        };
        let mut ordered: Vec<&orangu::gguf::TensorInfo> = file.tensors.iter().collect();
        quant::sort_for_quantizing(&mut ordered, |t| t.name.as_str());
        let inputs: Vec<quant::Tensor<'_>> = ordered
            .iter()
            .map(|t| quant::Tensor {
                name: &t.name,
                dims: &t.dims,
                source: t.ggml_type,
            })
            .collect();
        let plans = quant::plan_tensors(quant::Ftype::Q4KM, &inputs, &model);
        let map = unsafe { memmap2::Mmap::map(&std::fs::File::open(&path).unwrap()) }.unwrap();
        let types = [
            q::GGML_TYPE_Q3_K,
            q::GGML_TYPE_Q4_K,
            q::GGML_TYPE_Q5_K,
            q::GGML_TYPE_Q6_K,
        ];
        let mut by_role: BTreeMap<String, (usize, f64, [f64; 4])> = BTreeMap::new();
        for (t, plan) in ordered.iter().zip(plans) {
            let p = TensorPlan {
                name: t.name.clone(),
                dims: t.dims.clone(),
                ggml_type: plan.ggml_type,
            };
            if !adjustable(&p) || !(t.dims[0] as usize).is_multiple_of(256) {
                continue;
            }
            let elements = t.element_count() as usize;
            let ncols = t.dims[0] as usize;
            let bytes = quant::row_bytes(t.ggml_type, ncols) * (elements / ncols);
            let start = (file.data_offset + t.offset) as usize;
            let values = quant::decode(t.ggml_type, &map[start..start + bytes], elements).unwrap();
            let imp = matrix.matrix.get(&t.name);
            let err = |ty| auto::estimated_error(ty, &values, ncols, imp, auto::SAMPLE_ROWS);
            let entry = by_role.entry(role(&t.name)).or_default();
            entry.0 += 1;
            entry.1 += err(p.ggml_type);
            for (k, &ty) in types.iter().enumerate() {
                entry.2[k] += err(ty);
            }
        }
        println!(
            "{:>14} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "role", "n", "standard", "Q3_K", "Q4_K", "Q5_K", "Q6_K"
        );
        for (role, (n, standard, at)) in by_role {
            println!(
                "{role:>14} {n:>4} {standard:>10.5} {:>10.5} {:>10.5} {:>10.5} {:>10.5}",
                at[0], at[1], at[2], at[3]
            );
        }
    }
}

#[cfg(test)]
mod role_unit_tests {
    use super::*;

    #[test]
    fn every_block_s_tensor_of_a_kind_is_one_role() {
        assert_eq!(role("blk.3.ffn_down.weight"), "ffn_down");
        assert_eq!(role("blk.12.ffn_up_exps.weight"), "ffn_up_exps");
        assert_eq!(role("output.weight"), "output");
        assert_eq!(role("token_embd.weight"), "token_embd");
    }

    fn cost(ggml_type: u32, bytes: u64) -> Cost {
        Cost {
            ggml_type,
            bytes,
            error: 0.0,
        }
    }

    #[test]
    fn a_role_is_lowered_about_a_quarter() {
        let costs = [
            cost(q::GGML_TYPE_Q3_K, 110),
            cost(q::GGML_TYPE_IQ4_XS, 136),
            cost(q::GGML_TYPE_Q4_K, 144),
            cost(q::GGML_TYPE_Q5_K, 176),
            cost(q::GGML_TYPE_Q6_K, 210),
        ];
        let lowered_to = |t| lowered(&costs, t).map(|c| costs[c].ggml_type);
        assert_eq!(lowered_to(q::GGML_TYPE_Q4_K), Some(q::GGML_TYPE_Q3_K));
        assert_eq!(lowered_to(q::GGML_TYPE_Q6_K), Some(q::GGML_TYPE_Q4_K));
        assert_eq!(lowered_to(q::GGML_TYPE_Q5_K), Some(q::GGML_TYPE_Q3_K));
        assert_eq!(lowered_to(q::GGML_TYPE_Q3_K), None);
    }
}

#[cfg(test)]
mod group_tests {
    use super::*;

    fn tensor(types: &[(u32, u64, f64, f64)]) -> Tensor {
        Tensor {
            costs: types
                .iter()
                .map(|&(ggml_type, bytes, error, _)| Cost {
                    ggml_type,
                    bytes,
                    error,
                })
                .collect(),
            time_us: types
                .iter()
                .map(|&(.., us)| [[us; 2], [us / 100.0; 2]])
                .collect(),
            on_device: true,
        }
    }

    #[test]
    fn a_role_takes_the_types_all_its_blocks_can_at_their_sum() {
        let (k4, k6, q8) = (q::GGML_TYPE_Q4_K, q::GGML_TYPE_Q6_K, q::GGML_TYPE_Q8_0);
        let tensors = vec![
            tensor(&[(k4, 100, 2.0, 10.0), (k6, 150, 0.5, 13.0)]),
            tensor(&[
                (k4, 100, 3.0, 10.0),
                (k6, 150, 1.0, 13.0),
                (q8, 200, 0.1, 16.0),
            ]),
        ];
        let m = merged(&tensors, &[0, 1]);
        let types: Vec<u32> = m.costs.iter().map(|c| c.ggml_type).collect();
        assert_eq!(types, [k4, k6]);
        assert_eq!((m.costs[0].bytes, m.costs[0].error), (200, 5.0));
        assert_eq!(m.time_us[1][0][0], 26.0);

        let plans: Vec<TensorPlan> = [
            "blk.0.ffn_down.weight",
            "blk.1.ffn_down.weight",
            "output.weight",
        ]
        .iter()
        .map(|n| TensorPlan {
            name: n.to_string(),
            dims: vec![256, 256],
            ggml_type: k4,
        })
        .collect();
        let three = vec![tensors[0].clone(), tensors[1].clone(), tensors[0].clone()];
        assert_eq!(
            role_groups(&[0, 1, 2], &plans, &three, &[]),
            [vec![0, 1], vec![2]]
        );
        let pinned = [parse_override("blk\\.1\\.=q6_k").unwrap()];
        assert_eq!(
            role_groups(&[0, 1, 2], &plans, &three, &pinned),
            [vec![0], vec![1], vec![2]]
        );
    }
}
