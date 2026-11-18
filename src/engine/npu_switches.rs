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

//! The NPU switches the forward passes read: the prefill width the
//! feed-forward graphs are compiled for, and whether attention projections
//! may run on the device. The server's NPU tool measures and sets them.

/// ever compiled for.
///
/// A graph on this device has one static shape, so this is part of a
/// block's cache identity: a request at any other width finds nothing and
/// goes to the CPU or GPU as before.
pub const PREFILL_WIDTH: usize = 16;

/// The prefill width to compile for, `ORANGU_NPU_PREFILL_WIDTH` overriding
/// [`PREFILL_WIDTH`].
///
/// A measurement knob first: the device is not equally efficient at every
/// width, and 16 is what this engine's chunk sizer happens to issue rather
/// than what the hardware likes. One block of llama 3.2 3B `Q8_0`, per
/// token on the device:
///
/// | width | per token | rate |
/// |---|---|---|
/// | 1 | 3.27 ms | 46 GF/s |
/// | 16 | 0.342 ms | 441 GF/s |
/// | 32 | 0.415 ms | 364 GF/s |
/// | 64 | **0.256 ms** | **590 GF/s** |
/// | 128 | 0.281 ms | 537 GF/s |
///
/// Whether that 1.34x survives end to end is a different question, because
/// a wider graph only gets used if `prefill_in_chunks` issues chunks that
/// wide, and its sizer is bounded by `ORANGU_PREFILL_CHUNK_MS`. The two
/// have to move together, which is what this knob is for.
pub fn prefill_width() -> usize {
    if let Some(forced) = forced_prefill_width() {
        return forced;
    }
    match MEASURED_WIDTH.load(std::sync::atomic::Ordering::Relaxed) {
        0 => PREFILL_WIDTH,
        measured => measured,
    }
}

/// `ORANGU_NPU_PREFILL_WIDTH`, when an operator has pinned the width.
pub fn forced_prefill_width() -> Option<usize> {
    static FORCED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| {
        crate::engine::backend::tuning_profile::var("ORANGU_NPU_PREFILL_WIDTH")
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|w| *w > 1)
    })
}

/// What [`probe_prefill_width`] measured on this device, or `0` before it
/// has run.
static MEASURED_WIDTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Records the width [`prefill_width`] answers with when none is forced.
pub fn set_measured_width(width: usize) {
    MEASURED_WIDTH.store(width, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the attention projections may run on the device, on unless
/// `ORANGU_NPU_ATTN=0`.
///
/// A serving-time switch rather than a compile-time one, so one cache
/// serves both arms of a measurement and the two differ in nothing else.
pub fn attention_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_NPU_ATTN"))
}
