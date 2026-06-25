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

//! Criterion benchmarks for the compression module.
//!
//! Run with:
//!   cargo bench --bench compression
//!
//! Results are written to `target/criterion/`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use orangu::compression::{compress_shell_output, compress_shell_output_with_stats};

// ── Representative fixture generators ────────────────────────────────────────

fn cargo_build_large() -> String {
    let mut out = String::new();
    for i in 0..80 {
        out.push_str(&format!("   Compiling crate_{i} v0.1.{i}\n"));
    }
    out.push_str("error[E0308]: mismatched types\n");
    out.push_str("  --> src/main.rs:10:5\n");
    out.push_str("   |\n");
    out.push_str("10 |     let x: i32 = \"hello\";\n");
    out.push_str("   |             ^^^  expected i32, found &str\n");
    out.push_str("error: could not compile `my_crate` due to 1 previous error\n");
    out
}

fn cargo_test_mixed() -> String {
    let mut out = String::new();
    out.push_str("running 120 tests\n");
    for i in 0..118 {
        out.push_str(&format!("test module::test_{i:03} ... ok\n"));
    }
    out.push_str("test module::test_118 ... FAILED\n");
    out.push_str("test module::test_119 ... FAILED\n");
    out.push_str("failures:\n");
    out.push_str("---- module::test_118 stdout ----\n");
    out.push_str("thread 'main' panicked at 'assertion failed', src/lib.rs:42\n");
    out.push_str("test result: FAILED. 118 passed; 2 failed; 0 ignored\n");
    out
}

fn git_diff_large() -> String {
    let mut out = String::new();
    out.push_str("diff --git a/src/main.rs b/src/main.rs\n");
    out.push_str("index abc123..def456 100644\n");
    out.push_str("--- a/src/main.rs\n");
    out.push_str("+++ b/src/main.rs\n");
    out.push_str("@@ -1,10 +1,10 @@\n");
    for i in 0..600 {
        if i % 10 == 0 {
            out.push_str(&format!("+fn new_fn_{i}() {{\n"));
        } else {
            out.push_str(&format!(" fn existing_{i}() {{\n"));
        }
    }
    out
}

fn git_log_many() -> String {
    let mut out = String::new();
    for i in 0..50 {
        out.push_str(&format!(
            "commit {i:040x}\nAuthor: Dev <dev@example.com>\nDate:   Mon Jun 22 10:0{i} 2026\n\n    commit message {i}\n\n",
            i = i
        ));
    }
    out
}

fn short_output() -> String {
    "Finished dev [unoptimized] target(s) in 0.42s\n".repeat(5)
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_cargo_build(c: &mut Criterion) {
    let input = cargo_build_large();
    c.bench_function("compress cargo build (80 crates + error)", |b| {
        b.iter(|| compress_shell_output(black_box("cargo build"), black_box(&input), 20))
    });
}

fn bench_cargo_test(c: &mut Criterion) {
    let input = cargo_test_mixed();
    c.bench_function("compress cargo test (120 tests, 2 failed)", |b| {
        b.iter(|| compress_shell_output(black_box("cargo test"), black_box(&input), 20))
    });
}

fn bench_git_diff(c: &mut Criterion) {
    let input = git_diff_large();
    c.bench_function("compress git diff (600 lines)", |b| {
        b.iter(|| compress_shell_output(black_box("git diff HEAD"), black_box(&input), 20))
    });
}

fn bench_git_log(c: &mut Criterion) {
    let input = git_log_many();
    c.bench_function("compress git log (50 commits)", |b| {
        b.iter(|| compress_shell_output(black_box("git log"), black_box(&input), 20))
    });
}

fn bench_short_passthrough(c: &mut Criterion) {
    let input = short_output();
    c.bench_function("compress short output (passthrough)", |b| {
        b.iter(|| compress_shell_output(black_box("cargo build"), black_box(&input), 20))
    });
}

fn bench_with_stats(c: &mut Criterion) {
    let input = cargo_build_large();
    c.bench_function("compress_with_stats cargo build", |b| {
        b.iter(|| compress_shell_output_with_stats(black_box("cargo build"), black_box(&input), 20))
    });
}

criterion_group!(
    benches,
    bench_cargo_build,
    bench_cargo_test,
    bench_git_diff,
    bench_git_log,
    bench_short_passthrough,
    bench_with_stats,
);
criterion_main!(benches);
