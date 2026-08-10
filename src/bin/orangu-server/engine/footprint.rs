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

//! What this model will actually put on the device, against what the
//! device has — answered at startup, from the loaded model's own tensor
//! table and KV geometry.
//!
//! # Why this is not `engine::plan`
//!
//! [`crate::engine::plan`] answers "could this machine run that file at
//! all", from a GGUF nobody has opened, in terms of system RAM. This
//! answers a narrower and later question: the model *is* loaded, a device
//! *has* been chosen, and the useful thing to say is how much of that
//! device the weights take and how much context is left over.
//!
//! # Why it does not say yes or no
//!
//! It is tempting to end with a verdict, and the verdict would be wrong.
//! Weights reach the device lazily and are never evicted
//! (`VulkanBackend::weight_buffer`), the KV cache is allocated per request
//! at that request's own size rather than at the context limit, and the
//! transient arenas grow to whatever the widest prefill needed. A model
//! whose weights exceed VRAM still runs — the driver pages, and it is slow
//! rather than broken. So a hard "does not fit" refusal would turn working,
//! if slow, configurations into failures, and a cheerful "fits" would be a
//! claim about a KV cache nobody has sized yet.
//!
//! What is actually decidable, and is what this reports:
//!
//! - the **weights**, exactly, split into what goes to the device and what
//!   stays on the CPU (routed experts);
//! - the **headroom** left on the device after them;
//! - **how much context that headroom buys**, across the configured slots,
//!   which is the number an operator can act on;
//! - and a **warning** when the weights alone do not fit, which is the one
//!   case where the answer really is known in advance.

use crate::engine::backend::device::DeviceCandidate;
use crate::engine::backend::device_resident_split;
use crate::engine::backend::vulkan_shaders::KvStorage;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig};
use orangu::format::format_bytes;

/// Context granularity every per-token quantity is reported at. Bytes per
/// token is a number with too many zeroes in front of it to compare at a
/// glance; bytes per thousand tokens is the scale prompts are actually
/// discussed in.
const CONTEXT_STEP: usize = 1024;

/// What one model puts on one device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFootprint {
    /// Weight bytes a GPU backend uploads.
    pub weights_device_bytes: u64,
    /// Weight bytes that stay in host memory whatever the backend —
    /// routed-expert tensors, which have no GPU path (see
    /// `engine::backend::device_resident_split`).
    pub weights_host_bytes: u64,
    /// The GPU-side KV mirror for `CONTEXT_STEP` tokens of one sequence.
    pub kv_bytes_per_step: u64,
    /// How many sequences can be resident at once — `[orangu-server].slots`.
    /// Each carries its own KV cache.
    pub slots: usize,
    /// The context ceiling a single request can ask for, which is what
    /// `engine::generate` clamps every request's KV capacity to.
    pub n_ctx_train: usize,
    /// How the KV mirror is stored, which halves or doubles every number
    /// above it. `None` on a backend with no GPU KV mirror at all.
    pub kv_storage: Option<KvStorage>,
}

impl DeviceFootprint {
    /// Measures `model` against the KV geometry `probe` describes.
    ///
    /// `probe` is a `ModelForward::new_kv_cache(1)` — a cache built at one
    /// token, which allocates nothing worth counting and carries the full
    /// per-layer shape. `main` already builds one of these for the
    /// slot-persistence structure tag, so this asks nothing new of the
    /// caller.
    pub fn measure(
        model: &LoadedModel,
        probe: &KvCache,
        kv_storage: Option<KvStorage>,
        slots: usize,
    ) -> Self {
        let (weights_device_bytes, weights_host_bytes) =
            device_resident_split(model.tensor_sizes());
        let kv_bytes_per_step = kv_storage
            .map(|storage| probe.gpu_mirror_bytes(CONTEXT_STEP, model.config.n_head, storage))
            .unwrap_or(0);
        Self {
            weights_device_bytes,
            weights_host_bytes,
            kv_bytes_per_step,
            slots,
            n_ctx_train: model.config.n_ctx_train,
            kv_storage,
        }
    }

    /// What **one device of a split** holds: the weights placed on it and the
    /// KV mirror for the layers placed on it.
    ///
    /// `weights_device_bytes` is passed in rather than recomputed because the
    /// caller already has it from [`DeviceFootprint::weights_per_device`], and
    /// that function asks the model where each tensor actually went — a second
    /// derivation here could disagree with the first.
    ///
    /// `weights_host_bytes` is **zero** on every device of a split, not the
    /// model's host total. Routed experts live in system RAM and are on no
    /// device at all; charging them to each card in turn would count them
    /// twice on a two-card split and make every headroom figure wrong in the
    /// same direction. The split report states the host total once, where it
    /// belongs.
    pub fn for_split_device(
        config: &ModelConfig,
        probe: &KvCache,
        kv_storage: Option<KvStorage>,
        slots: usize,
        weights_device_bytes: u64,
        layer_device: &[usize],
        device: usize,
    ) -> Self {
        let kv_bytes_per_step = kv_storage
            .map(|storage| {
                probe.gpu_mirror_bytes_where(
                    CONTEXT_STEP,
                    config.n_head,
                    storage,
                    // A layer the plan does not mention is on the head device,
                    // which is where `LoadedModel::device_for_tensor` puts
                    // anything it has no placement for.
                    |layer| layer_device.get(layer).copied().unwrap_or(0) == device,
                )
            })
            .unwrap_or(0);
        Self {
            weights_device_bytes,
            weights_host_bytes: 0,
            kv_bytes_per_step,
            slots,
            n_ctx_train: config.n_ctx_train,
            kv_storage,
        }
    }

    /// Device-resident weight bytes per device, once a split plan has been
    /// stamped onto `model`.
    ///
    /// Asks the model itself which device each tensor went to
    /// (`LoadedModel::device_for_tensor`) rather than re-deriving it from
    /// the plan: that is the same call `LoadedModel::matrix` makes when it
    /// stamps a `QuantMatrix`, so the report cannot disagree with where the
    /// weights actually went — which is the failure mode a second copy of
    /// the rule would eventually produce.
    pub fn weights_per_device(model: &LoadedModel, n_devices: usize) -> Vec<u64> {
        let mut per_device = vec![0u64; n_devices.max(1)];
        for (name, bytes) in model.tensor_sizes() {
            let (device_bytes, _) = device_resident_split(std::iter::once((name, bytes)));
            if device_bytes == 0 {
                continue;
            }
            let device = model.device_for_tensor(name).min(per_device.len() - 1);
            per_device[device] += device_bytes;
        }
        per_device
    }

    /// Device-resident weight bytes per transformer layer.
    ///
    /// What a fill-in-order placement needs and a proportional one does
    /// not: "how many of these layers fit in 4 GiB" is only answerable per
    /// layer, and layers are not all the same size (a model with per-layer
    /// expert counts or varying FFN widths differs by a factor across its
    /// own depth).
    ///
    /// Everything outside a numbered `blk.<n>.` block — embeddings, the
    /// output norm, `lm_head` — is excluded rather than spread: those go to
    /// the first device by `LoadedModel::device_for_tensor`'s own rule, and
    /// charging them to a layer would misplace the boundary by their size.
    pub fn weights_per_layer(model: &LoadedModel, n_layer: usize) -> Vec<u64> {
        let mut per_layer = vec![0u64; n_layer];
        for (name, bytes) in model.tensor_sizes() {
            let (device_bytes, _) = device_resident_split(std::iter::once((name, bytes)));
            if device_bytes == 0 {
                continue;
            }
            let Some(layer) = name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|digits| digits.parse::<usize>().ok())
                .filter(|layer| *layer < n_layer)
            else {
                continue;
            };
            per_layer[layer] += device_bytes;
        }
        per_layer
    }

    /// How many tokens of context, across every slot, `headroom` bytes buy.
    ///
    /// `None` when this backend has no GPU KV mirror, and `Some(0)` when
    /// there is no headroom — which are different answers and must not be
    /// collapsed into one.
    pub fn kv_tokens_in(&self, headroom: u64) -> Option<usize> {
        let per_step = self.kv_bytes_per_step.checked_mul(self.slots as u64)?;
        if per_step == 0 {
            return None;
        }
        Some((headroom / per_step * CONTEXT_STEP as u64) as usize)
    }

    /// The device's capacity less the weights, or `None` when the capacity
    /// is unknown — which is not zero, and must not be reported as a
    /// shortfall (see `engine::backend::device`'s own note on unknown VRAM).
    pub fn headroom_on(&self, device: &DeviceCandidate) -> Option<u64> {
        self.headroom_in(device.vram_total_bytes)
    }

    /// The same against a bare capacity, for a device of a split — where
    /// there is no `DeviceCandidate` left to ask (`Backend::as_wgpu` is
    /// `None` on the multi-device wrapper by design, so the split report
    /// carries the capacities out instead).
    pub fn headroom_in(&self, total_bytes: Option<u64>) -> Option<u64> {
        Some(total_bytes?.saturating_sub(self.weights_device_bytes))
    }

    /// `Some(shortfall)` when the weights alone exceed the device, `None`
    /// when they fit or the capacity is unknown.
    ///
    /// The one thing here that is knowable in advance and worth saying
    /// loudly: past this point the driver is paging weights in and out of
    /// VRAM on every token, which is a large, silent slowdown that reads
    /// like "orangu is slow on this card" rather than like a capacity
    /// problem.
    pub fn shortfall_on(&self, device: &DeviceCandidate) -> Option<u64> {
        self.shortfall_in(device.vram_total_bytes)
    }

    /// [`DeviceFootprint::shortfall_on`] against a bare capacity, for a
    /// device of a split.
    pub fn shortfall_in(&self, total_bytes: Option<u64>) -> Option<u64> {
        let total = total_bytes?;
        (self.weights_device_bytes > total).then(|| self.weights_device_bytes - total)
    }

    /// The startup lines for one device, or empty when there is nothing
    /// device-specific to say (a CPU-only run).
    pub fn report(&self, api: &str, device: &DeviceCandidate) -> Vec<String> {
        let mut lines = Vec::new();
        let mut weights = format!(
            "weights {} on device",
            format_bytes(self.weights_device_bytes)
        );
        if self.weights_host_bytes > 0 {
            // Named, not just totalled: "the rest is on the CPU" invites
            // the question of whether that is a fallback or a bug, and the
            // answer (routed experts have no GPU path at all) is a
            // property of the engine rather than of this machine.
            weights.push_str(&format!(
                ", {} in host memory (routed experts)",
                format_bytes(self.weights_host_bytes)
            ));
        }
        lines.push(format!("orangu-server: [{api}] {weights}"));

        match (device.vram_total_bytes, self.kv_storage) {
            (Some(total), storage) => {
                let headroom = self.headroom_on(device).unwrap_or(0);
                let mut line = format!(
                    "orangu-server: [{api}] {} of {} used by weights, {} free",
                    format_bytes(self.weights_device_bytes),
                    format_bytes(total),
                    format_bytes(headroom)
                );
                if let (Some(tokens), Some(storage)) = (self.kv_tokens_in(headroom), storage) {
                    // Capped at what a request can actually ask for: a card
                    // with room for a million tokens of KV and a model
                    // trained on 8k has room for 8k, and saying otherwise
                    // invites someone to go looking for the missing
                    // context.
                    let usable = tokens.min(self.n_ctx_train.saturating_mul(self.slots));
                    line.push_str(&format!(
                        " — room for about {} tokens of {:?} KV across {} slot{}",
                        usable,
                        storage,
                        self.slots,
                        if self.slots == 1 { "" } else { "s" }
                    ));
                }
                lines.push(line);
            }
            // A device whose capacity the API declined to report. Saying so
            // is better than silence: it explains why no headroom line
            // follows, and it is the same "unknown is not zero" rule the
            // ranking policy applies.
            (None, _) => lines.push(format!(
                "orangu-server: [{api}] this device does not report its memory size, \
                 so there is no headroom figure for it"
            )),
        }

        if let Some(shortfall) = self.shortfall_on(device) {
            lines.push(format!(
                "orangu-server: [{api}] the weights are {} larger than this device — the \
                 driver will page them in and out on every token, which is slow rather \
                 than fatal. A smaller quantization, or `device`/`backend = cpu`, avoids it.",
                format_bytes(shortfall)
            ));
        }
        lines
    }

    /// The same numbers for `/props`, beside the tuning report a benchmark
    /// result already carries.
    pub fn to_json(&self, device: &DeviceCandidate) -> serde_json::Value {
        self.to_json_in(device.vram_total_bytes)
    }

    /// [`DeviceFootprint::to_json`] against a bare capacity, for one device of
    /// a split — the same field names, so a reader (and `orangu-bench`) parses
    /// one shape whether the run used one card or four.
    pub fn to_json_in(&self, total_bytes: Option<u64>) -> serde_json::Value {
        serde_json::json!({
            "weights_device_bytes": self.weights_device_bytes,
            "weights_host_bytes": self.weights_host_bytes,
            "kv_bytes_per_1k_tokens_per_slot": self.kv_bytes_per_step,
            "slots": self.slots,
            "device_total_bytes": total_bytes,
            "headroom_bytes": self.headroom_in(total_bytes),
            "shortfall_bytes": self.shortfall_in(total_bytes),
            "kv_tokens_in_headroom": self
                .headroom_in(total_bytes)
                .and_then(|headroom| self.kv_tokens_in(headroom)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::device::DeviceClass;

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    fn card(vram: Option<u64>) -> DeviceCandidate {
        DeviceCandidate {
            index: 0,
            name: "test card".to_string(),
            class: DeviceClass::Discrete,
            vram_total_bytes: vram,
            id: None,
            driver: None,
        }
    }

    fn footprint(weights: u64, kv_per_step: u64, slots: usize) -> DeviceFootprint {
        DeviceFootprint {
            weights_device_bytes: weights,
            weights_host_bytes: 0,
            kv_bytes_per_step: kv_per_step,
            slots,
            n_ctx_train: 1_000_000,
            kv_storage: Some(KvStorage::F16),
        }
    }

    #[test]
    fn headroom_is_the_capacity_less_the_weights() {
        let f = footprint(3 * GIB, MIB, 1);
        assert_eq!(f.headroom_on(&card(Some(4 * GIB))), Some(GIB));
        assert_eq!(f.shortfall_on(&card(Some(4 * GIB))), None);
    }

    /// Over-capacity is a shortfall, and headroom clamps at zero rather
    /// than wrapping — the subtraction is on unsigned bytes.
    #[test]
    fn weights_larger_than_the_card_are_a_shortfall_not_negative_headroom() {
        let f = footprint(6 * GIB, MIB, 1);
        assert_eq!(f.headroom_on(&card(Some(4 * GIB))), Some(0));
        assert_eq!(f.shortfall_on(&card(Some(4 * GIB))), Some(2 * GIB));
    }

    /// Unknown capacity is unknown, in both directions: no headroom figure
    /// and — crucially — no shortfall claimed against a device that never
    /// said how big it was.
    #[test]
    fn an_unknown_capacity_yields_neither_headroom_nor_a_shortfall() {
        let f = footprint(6 * GIB, MIB, 1);
        assert_eq!(f.headroom_on(&card(None)), None);
        assert_eq!(f.shortfall_on(&card(None)), None);
    }

    /// A ModelConfig is all [`DeviceFootprint::for_split_device`] needs of a
    /// model — the head count for the KV geometry and the trained context —
    /// which is why it takes one rather than a `LoadedModel`: the weights and
    /// the placement are passed in by the caller that already has them.
    fn config(n_head: usize, n_ctx_train: usize) -> ModelConfig {
        ModelConfig {
            architecture: "llama".to_string(),
            n_vocab: 0,
            n_embd: 0,
            n_layer: 4,
            n_head,
            n_head_kv: n_head,
            head_dim: 1,
            n_ctx_train,
            rope_dim: 0,
            rope_freq_base: 0.0,
            rms_eps: 0.0,
            pooling_type: crate::engine::loader::PoolingType::Mean,
        }
    }

    /// Each device of a split is charged for its own layers' KV and nothing
    /// else, and the parts add up to the whole. A footprint that gave every
    /// device the model's full KV figure would report the *same* headroom
    /// pressure on a card holding two layers as on one holding fourteen —
    /// which is exactly the case an operator splits a model to get out of.
    #[test]
    fn a_split_device_is_charged_for_its_own_layers_only() {
        let probe = KvCache::new_with_dims(1, &[64, 64, 256, 256]);
        let config = config(8, 4096);
        // Layers 0 and 1 on device 0, layers 2 and 3 on device 1.
        let layer_device = [0, 0, 1, 1];
        let on = |device: usize| {
            DeviceFootprint::for_split_device(
                &config,
                &probe,
                Some(KvStorage::F16),
                1,
                0,
                &layer_device,
                device,
            )
        };
        let whole = DeviceFootprint {
            kv_bytes_per_step: probe.gpu_mirror_bytes(CONTEXT_STEP, 8, KvStorage::F16),
            ..footprint(0, 0, 1)
        };
        assert_eq!(
            on(0).kv_bytes_per_step + on(1).kv_bytes_per_step,
            whole.kv_bytes_per_step,
            "the devices' shares must add up to the model's"
        );
        // And they are not equal shares: these layers are not equal sizes.
        assert!(on(1).kv_bytes_per_step > on(0).kv_bytes_per_step * 3);
        // A device the plan gave nothing to is charged nothing — not the
        // whole model.
        assert_eq!(on(2).kv_bytes_per_step, 0);
    }

    /// Routed experts are in system RAM and on no device. Charging them to
    /// each device in turn would count them once per card and pull every
    /// headroom figure down by the same wrong amount.
    #[test]
    fn host_weights_are_not_charged_to_any_device_of_a_split() {
        let probe = KvCache::new_with_dims(1, &[64]);
        let f = DeviceFootprint::for_split_device(
            &config(8, 4096),
            &probe,
            Some(KvStorage::F16),
            1,
            3 * GIB,
            &[0],
            0,
        );
        assert_eq!(f.weights_host_bytes, 0);
        assert_eq!(f.weights_device_bytes, 3 * GIB);
        // The capacity-taking core answers exactly as the DeviceCandidate one
        // does — a split has no candidate left to ask, and the two must not
        // drift into different arithmetic.
        assert_eq!(
            f.headroom_in(Some(4 * GIB)),
            f.headroom_on(&card(Some(4 * GIB)))
        );
        assert_eq!(f.shortfall_in(Some(2 * GIB)), Some(GIB));
        assert_eq!(f.headroom_in(None), None);
        assert_eq!(
            f.to_json_in(Some(4 * GIB))["headroom_bytes"],
            serde_json::json!(GIB)
        );
    }

    /// A device with no GPU KV mirror — the host overflow tier of
    /// `device_split = cpu` — is charged no KV, and says so as `None` rather
    /// than as zero bytes of a mirror it does not have.
    #[test]
    fn the_host_tier_of_a_split_has_no_kv_mirror() {
        let probe = KvCache::new_with_dims(1, &[64, 64]);
        let f =
            DeviceFootprint::for_split_device(&config(8, 4096), &probe, None, 1, GIB, &[0, 1], 1);
        assert_eq!(f.kv_bytes_per_step, 0);
        assert_eq!(f.kv_storage, None);
        assert_eq!(f.kv_tokens_in(GIB), None);
    }

    #[test]
    fn kv_tokens_scale_down_with_the_slot_count() {
        // 1 MiB per 1k tokens per slot, 1 GiB of headroom -> 1024k tokens
        // on one slot, a quarter of that when four caches share the space.
        let one = footprint(0, MIB, 1);
        let four = footprint(0, MIB, 4);
        assert_eq!(one.kv_tokens_in(GIB), Some(1024 * CONTEXT_STEP));
        assert_eq!(four.kv_tokens_in(GIB), Some(256 * CONTEXT_STEP));
    }

    /// No headroom is `Some(0)` tokens; no GPU KV mirror at all is `None`.
    /// Two different situations, two different answers.
    #[test]
    fn no_headroom_and_no_kv_mirror_are_different_answers() {
        assert_eq!(footprint(0, MIB, 1).kv_tokens_in(0), Some(0));
        let mut cpu = footprint(0, 0, 1);
        cpu.kv_storage = None;
        assert_eq!(cpu.kv_tokens_in(GIB), None);
    }

    /// The shortfall warning has to name the amount and a way out; a
    /// warning that only says "too big" is one an operator cannot act on.
    #[test]
    fn the_report_warns_with_the_shortfall_when_the_weights_do_not_fit() {
        let f = footprint(6 * GIB, MIB, 1);
        let lines = f.report("vulkan", &card(Some(4 * GIB))).join("\n");
        assert!(lines.contains("2.00 GiB larger"), "{lines}");
        assert!(lines.contains("quantization"), "{lines}");
    }

    #[test]
    fn the_report_names_host_resident_weights_when_there_are_any() {
        let mut f = footprint(GIB, MIB, 1);
        f.weights_host_bytes = 3 * GIB;
        let lines = f.report("vulkan", &card(Some(4 * GIB))).join("\n");
        assert!(lines.contains("3.00 GiB in host memory"), "{lines}");
        assert!(lines.contains("routed experts"), "{lines}");
        // And they must not be counted against the card.
        assert!(!lines.contains("larger than this device"), "{lines}");
    }

    /// A card with more headroom than the model can use must not advertise
    /// context the engine will refuse to allocate.
    #[test]
    fn the_reported_context_is_capped_at_what_a_request_can_ask_for() {
        let mut f = footprint(0, MIB, 1);
        f.n_ctx_train = 4096;
        let lines = f.report("vulkan", &card(Some(64 * GIB))).join("\n");
        assert!(lines.contains("about 4096 tokens"), "{lines}");
    }

    #[test]
    fn a_device_that_reports_no_memory_says_so_rather_than_claiming_zero() {
        let f = footprint(2 * GIB, MIB, 1);
        let lines = f.report("vulkan", &card(None)).join("\n");
        assert!(lines.contains("does not report its memory size"), "{lines}");
        assert!(!lines.contains("free"), "{lines}");
    }
}
