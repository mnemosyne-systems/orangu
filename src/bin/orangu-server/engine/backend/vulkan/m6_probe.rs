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

//! Does the GPU actually beat eight CPU cores on an **expert-shaped** matmul?
//!
//! `BIG.md`'s M6 proposes routing MoE routed experts through this backend. The
//! plumbing that needs — a transient upload path, because both the weight
//! arena and the op cache are keyed on `QuantMatrix::cache_key()` and would
//! otherwise grow once per expert — is several hundred lines in a file whose
//! failure mode is device loss. This measures the premise first.
//!
//! The shape is `gemma-4-26B-A4B`'s own: `ffn_gate_up_exps` is
//! `[2816 in, 1408 out]` per expert at `Q4_K`, and at prefill the batch-union
//! hands one expert the rows of many tokens at once. If the GPU does not win
//! that decisively, no amount of plumbing makes M6 worth having.
//!
//! `cargo test --release --bin orangu-server m6_expert_shaped -- --ignored --nocapture`

use super::*;
use crate::engine::backend::{Backend, CpuBackend};
use crate::engine::loader::test_quant_matrix;
use crate::engine::quant::GGML_TYPE_Q4_K;

#[test]
#[ignore]
fn m6_expert_shaped_matmul_gpu_versus_cpu() {
    let Some(gpu) = shared_test_backend() else {
        eprintln!("skipping: no GPU backend");
        return;
    };
    const IN_DIM: usize = 2816;
    const OUT_DIM: usize = 1408;
    // Q4_K: 144 bytes per 256 elements.
    let bytes = vec![0x42u8; IN_DIM * OUT_DIM / 256 * 144];
    let w = test_quant_matrix(&bytes, GGML_TYPE_Q4_K, IN_DIM, OUT_DIM);
    let cpu = CpuBackend;

    println!("expert-shaped matmul: [{IN_DIM} x {OUT_DIM}] Q4_K");
    println!(
        "{:>8} {:>12} {:>12} {:>10}",
        "tokens", "CPU ms", "GPU ms", "speedup"
    );
    for &n in &[1usize, 8, 32, 128, 512] {
        let x = vec![0.05f32; IN_DIM * n];
        // Warm both paths: the GPU's first call builds its pipeline and
        // uploads the weight, which is exactly the cost M6 would pay per
        // expert and is measured separately below.
        let _ = cpu.matmul(&x, n, &w);
        let _ = gpu.matmul(&x, n, &w);

        let t = std::time::Instant::now();
        let a = cpu.matmul(&x, n, &w);
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = std::time::Instant::now();
        let b = gpu.matmul(&x, n, &w);
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(a.len(), b.len());
        println!(
            "{n:>8} {cpu_ms:>12.2} {gpu_ms:>12.2} {:>9.2}x",
            cpu_ms / gpu_ms.max(1e-9)
        );
    }

    // The other half of M6's cost: uploading one expert's weights. The
    // measurement above excludes it, because the weight is already
    // resident after the warmup — M6 would pay it once per expert per
    // layer call.
    println!("\nupload cost (what M6 pays per expert, excluded above):");
    let mib = bytes.len() as f64 / (1024.0 * 1024.0);
    println!("  one expert's gate_up slice = {mib:.2} MiB");
}
