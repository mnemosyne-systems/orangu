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

//! One run, in one file, readable on a machine that has never seen the machine
//! it came from — **profile there, analyze here**.
//!
//! # The problem this exists for
//!
//! `--history` already carries measurements between machines, but a history row
//! is eight columns: date, label, mode, n, best, mean, sd, sd_sample. It says a number was
//! 163.3 and nothing whatsoever about what produced it. That is fine while
//! every row comes off the same desk. It stops being fine the moment the
//! interesting comparison is against a GPU nobody local can boot — a different
//! API, a different shared-memory ceiling, a different set of kernels that
//! feature negotiation happened to turn on. Then "163.3 against 149.1" is not a
//! result, because the two runs may not have been running the same kernels at
//! all, and the row cannot say.
//!
//! `--flamegraph` has the opposite shape and the same gap: it produces a
//! `.folded` designed to be carried off the machine and re-read here
//! (`profile::read_profiles`), but it needs `perf`, which is Linux-only. The
//! device this is most needed for is the one that cannot produce one.
//!
//! So: a bundle is the whole run — every measurement, plus the configuration
//! that produced it, plus the environment it ran in — as a single JSON file to
//! copy back. [`read`] then reconstructs enough to render a chart and, more to
//! the point, to *diff two configurations against each other* ([`diff`]) so a
//! difference in throughput can be attributed to a difference in setup rather
//! than assumed to be one.
//!
//! # Why JSON and not the history TSV
//!
//! The configuration half is a nested, open-ended document that grows a field
//! whenever the engine grows a flag (`VulkanBackend::tuning_report`). Flattening
//! that into columns would either freeze the schema or make every reader parse a
//! moving target. JSON keeps [`diff`] schema-agnostic: it walks whatever is
//! there and reports the leaves that differ, so a bundle written by a build that
//! reports a flag this one has never heard of still compares correctly.
//!
//! [`SCHEMA`] is versioned so a future reader can tell an old bundle from a
//! malformed one, and [`read`] refuses a *newer* major schema rather than
//! silently reading fields that have moved.

use std::collections::BTreeMap;
use std::path::Path;

use crate::history::Record;

/// Bundle schema version. Bump the major when a field moves or changes meaning
/// — [`read`] refuses anything with a higher major than it knows, because a
/// silent misread of a benchmark archive is worse than a clear refusal.
pub const SCHEMA: u32 = 1;

/// A run, read back. The measurement half is [`Record`]s so the chart and the
/// history file take it unchanged; the configuration half stays as raw JSON so
/// [`diff`] can walk it without this file knowing the engine's flag list.
#[derive(Debug)]
pub struct Bundle {
    /// Where it was read from — the name used in tables and diffs.
    pub name: String,
    /// `YYYY-MM-DD` the run was taken, from the records.
    pub date: String,
    /// The server's `/props`, including its `gpu` block when it has one.
    pub props: serde_json::Value,
    /// Host facts the server does not report: OS, CPU, and (where readable)
    /// GPU clock state.
    pub host: serde_json::Value,
    /// How the benchmark itself was invoked.
    pub run: serde_json::Value,
    /// The GPU timestamp breakdown for the measured window, or `Null`.
    pub gpu_timings: serde_json::Value,
    /// The harness build that took the measurement — `orangu-bench 1.2.0
    /// (52c0443)`. Kept because a report rebuilt from this file must credit
    /// the tool that *measured* it, not the one drawing the page. `None` for a
    /// bundle written before the field carried a commit.
    pub tool: Option<String>,
    /// When it was measured, `2026-08-04T09:34:21Z`. `None` for a bundle
    /// written before the field existed — those carry only [`Bundle::date`].
    pub measured_at: Option<String>,
    pub records: Vec<Record>,
}

impl Bundle {
    /// The label the run's records carry, for a chart legend. Records within
    /// one bundle share a label by construction (`--label`, else the model),
    /// so the first one is the bundle's.
    pub fn label(&self) -> &str {
        self.records.first().map_or(&self.name, |r| &r.label)
    }

    /// `(mode, n) -> mean tok/s`, for the side-by-side table.
    pub fn by_point(&self) -> BTreeMap<(String, u32), f64> {
        self.records
            .iter()
            .map(|r| ((r.mode.clone(), r.n), r.mean))
            .collect()
    }
}

/// Write one run out.
///
/// `props` is passed in verbatim rather than re-fetched: it must be the
/// configuration that was live *while measuring*. Re-reading it afterwards
/// would usually agree and would occasionally, silently, not — a server
/// restarted mid-sweep is exactly the accident a bundle exists to make visible.
pub fn write(
    path: &str,
    props: &serde_json::Value,
    host: serde_json::Value,
    run: serde_json::Value,
    gpu_timings: &serde_json::Value,
    records: &[Record],
) -> anyhow::Result<()> {
    let doc = serde_json::json!({
        "schema": SCHEMA,
        // The harness's own build, not just its version. The measurement is
        // as much a product of the tool as of the server: this file exists
        // because "which build produced this number" has to be answerable
        // months later, and that question has two halves — `props.version`
        // and `props.commit` are the other one.
        "tool": format!("orangu-bench {}", orangu::build_info::id()),
        // *When*, to the second and in UTC. The records carry a date, which
        // cannot separate two runs taken on the same afternoon — which is
        // precisely when an A/B is taken.
        "measured_at": crate::history::now_utc(),
        "props": props,
        "host": host,
        "run": run,
        // Where the GPU's time went during the measured window, when the
        // server reports it. The one profiling instrument that works on a
        // platform without `perf` — see `report_gpu_timings`.
        "gpu_timings": gpu_timings,
        "records": records.iter().map(|r| serde_json::json!({
            "date": r.date,
            "label": r.label,
            "mode": r.mode,
            "n": r.n,
            "best": r.best,
            "mean": r.mean,
            "sd": r.sd,
            // Both estimators: `sd` is what this file has always carried
            // (population, ÷ n), `sd_sample` is the standard one (÷ n-1) that
            // a `±` from another benchmark can be put beside. `null` where one
            // repetition leaves it undefined.
            "sd_sample": r.sd_sample,
        })).collect::<Vec<_>>(),
    });
    // Pretty-printed on purpose: a bundle's whole job is to be read months
    // later, often in a diff or a code review, by someone who does not have
    // this tool to hand.
    std::fs::write(path, serde_json::to_string_pretty(&doc)? + "\n")?;
    Ok(())
}

/// Read one back.
pub fn read(path: &str) -> anyhow::Result<Bundle> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{path}: not a bundle ({e})"))?;
    let schema = doc.get("schema").and_then(serde_json::Value::as_u64);
    match schema {
        Some(v) if v as u32 <= SCHEMA => {}
        Some(v) => anyhow::bail!(
            "{path}: schema {v} is newer than this build understands ({SCHEMA}) — \
             read it with the orangu-bench that wrote it"
        ),
        None => anyhow::bail!("{path}: no schema field; this is not an orangu-bench bundle"),
    }
    let records = doc
        .get("records")
        .and_then(serde_json::Value::as_array)
        .map(|rows| rows.iter().filter_map(record_from).collect::<Vec<_>>())
        .unwrap_or_default();
    let name = Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    Ok(Bundle {
        date: records
            .first()
            .map_or_else(|| "?".to_string(), |r| r.date.clone()),
        name,
        props: doc.get("props").cloned().unwrap_or(serde_json::Value::Null),
        host: doc.get("host").cloned().unwrap_or(serde_json::Value::Null),
        run: doc.get("run").cloned().unwrap_or(serde_json::Value::Null),
        gpu_timings: doc
            .get("gpu_timings")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        tool: doc.get("tool").and_then(|v| v.as_str()).map(str::to_string),
        measured_at: doc
            .get("measured_at")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        records,
    })
}

/// A row that does not parse is dropped rather than failing the read, matching
/// `history::read`'s own rule and for the same reason: one bad row should not
/// cost a reader every good one around it.
fn record_from(v: &serde_json::Value) -> Option<Record> {
    let s = |k: &str| v.get(k)?.as_str().map(str::to_string);
    let f = |k: &str| v.get(k)?.as_f64();
    Some(Record {
        date: s("date")?,
        label: s("label")?,
        mode: s("mode")?,
        n: v.get("n")?.as_u64()? as u32,
        best: f("best")?,
        mean: f("mean")?,
        sd: f("sd")?,
        // Absent from a bundle written before this field existed, which is
        // read back as "not recorded" rather than as zero.
        sd_sample: f("sd_sample"),
    })
}

/// Every leaf of `props`/`host` where two bundles disagree, as
/// `(path, left, right)` with the JSON pointer flattened to dots.
///
/// **The reason to keep a bundle at all.** Two runs differing by 10% are only
/// evidence of anything once you know what else differed between them, and on
/// a `wgpu` engine the answer is routinely "a flag neither of us set on
/// purpose" — an adapter that declined a feature, an `ORANGU_*` left over in
/// one shell, a different device limit that silenced a kernel. Reading it off
/// two pretty-printed JSON documents by eye does not work; there are ~40 fields
/// and most of them agree.
///
/// Fields that *must* differ between two machines and say nothing about the
/// comparison — the adapter's name, uptime, the process id — are skipped, so
/// what is left is signal. A leaf present on one side and absent on the other
/// is reported too, with `—` for the missing side: a bundle written by an older
/// build not knowing about a flag is itself worth seeing.
pub fn diff(a: &Bundle, b: &Bundle) -> Vec<(String, String, String)> {
    /// Fields whose difference is not a finding.
    const IGNORE: &[&str] = &[
        "props.uptime_seconds",
        "props.pid",
        "props.workspace",
        "props.chat_template",
        "host.clocks",
    ];
    let mut left = BTreeMap::new();
    let mut right = BTreeMap::new();
    flatten("props", &a.props, &mut left);
    flatten("host", &a.host, &mut left);
    flatten("run", &a.run, &mut left);
    flatten("props", &b.props, &mut right);
    flatten("host", &b.host, &mut right);
    flatten("run", &b.run, &mut right);
    let mut keys: Vec<&String> = left.keys().chain(right.keys()).collect();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter()
        .filter(|k| !IGNORE.iter().any(|ig| k.starts_with(ig)))
        .filter_map(|k| {
            let (l, r) = (left.get(k), right.get(k));
            (l != r).then(|| {
                let show = |v: Option<&String>| v.cloned().unwrap_or_else(|| "—".to_string());
                (k.clone(), show(l), show(r))
            })
        })
        .collect()
}

/// `{"a": {"b": 1}}` → `{"prefix.a.b": "1"}`. Arrays are rendered whole rather
/// than indexed: nothing in a bundle uses an array as a record with meaningful
/// positions, and `gpus.0.sclk` differing is less readable than the list.
fn flatten(prefix: &str, v: &serde_json::Value, out: &mut BTreeMap<String, String>) {
    match v {
        serde_json::Value::Object(map) => {
            for (k, child) in map {
                flatten(&format!("{prefix}.{k}"), child, out);
            }
        }
        serde_json::Value::Null => {}
        serde_json::Value::String(s) => {
            out.insert(prefix.to_string(), s.clone());
        }
        other => {
            out.insert(prefix.to_string(), other.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(mode: &str, n: u32, mean: f64) -> Record {
        Record {
            date: "2026-07-31".into(),
            label: "test".into(),
            mode: mode.into(),
            n,
            best: mean + 1.0,
            mean,
            sd: 0.5,
            sd_sample: Some(0.61),
        }
    }

    /// A directory name no other test in this process will pick.
    ///
    /// A counter rather than the address of the caller's data, which is what
    /// this used to be: `{:p}` on a *slice* reference renders as
    /// `Pointer { addr: 0x…, metadata: 3 }` — braces, spaces and a colon.
    /// POSIX accepts all of those in a filename, so it worked locally and
    /// failed on Windows with `InvalidFilename`. A pointer was never a
    /// sensible thing to name a directory after anyway.
    fn unique_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        std::env::temp_dir().join(format!(
            "orangu-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn round_trip(props: serde_json::Value, records: &[Record]) -> Bundle {
        let dir = unique_dir("bundle-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("b.json");
        let p = path.to_string_lossy().into_owned();
        write(
            &p,
            &props,
            serde_json::json!({"os": "linux"}),
            serde_json::json!({"pp": [1024]}),
            &serde_json::json!({"steps": 3, "per_step_ms": {"total": 30.0}}),
            records,
        )
        .unwrap();
        let b = read(&p).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        b
    }

    /// The measurements have to survive the trip intact — this is an archive
    /// format, and a run that cannot be re-read is a run that was not kept.
    #[test]
    fn a_written_bundle_reads_back_with_its_measurements() {
        let records = vec![rec("pp", 1024, 163.3), rec("tg", 0, 47.5)];
        let got = round_trip(serde_json::json!({"backend": "Vulkan/x"}), &records);
        assert_eq!(got.records, records);
        assert_eq!(got.date, "2026-07-31");
        assert_eq!(got.label(), "test");
        assert_eq!(got.props["backend"], "Vulkan/x");
    }

    /// A bundle from a build that knows more than this one does must be
    /// refused, not half-read. Silently misreading an archive is the failure
    /// mode a version field exists to prevent.
    #[test]
    fn a_newer_schema_is_refused_rather_than_misread() {
        let dir = unique_dir("bundle-newer");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("b.json");
        let p = path.to_string_lossy().into_owned();
        std::fs::write(&path, format!(r#"{{"schema": {}}}"#, SCHEMA + 1)).unwrap();
        let err = read(&p).expect_err("a newer schema must be refused");
        assert!(err.to_string().contains("newer"), "{err}");
        // ...and a file that is not a bundle at all says so, rather than
        // reading as an empty run.
        std::fs::write(&path, "{}").unwrap();
        let err = read(&p).expect_err("a schema-less document must be refused");
        assert!(
            err.to_string().contains("not an orangu-bench bundle"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The diff has to find a nested flag that differs, ignore the fields that
    /// always differ between two machines, and report a field one side simply
    /// does not have.
    #[test]
    fn the_diff_reports_configuration_and_skips_the_noise() {
        let records = vec![rec("pp", 1024, 163.3)];
        let a = round_trip(
            serde_json::json!({
                "pid": 1,
                "uptime_seconds": 10,
                "gpu": {"flags": {"coop_vec4_tile_w": true, "kv_storage": "F16"}},
            }),
            &records,
        );
        let b = round_trip(
            serde_json::json!({
                "pid": 2,
                "uptime_seconds": 999,
                "gpu": {"flags": {"coop_vec4_tile_w": false}},
            }),
            &records,
        );
        let found = diff(&a, &b);
        let keys: Vec<&str> = found.iter().map(|(k, _, _)| k.as_str()).collect();
        assert!(
            keys.contains(&"props.gpu.flags.coop_vec4_tile_w"),
            "the flag that differs must be reported: {keys:?}"
        );
        assert!(
            keys.contains(&"props.gpu.flags.kv_storage"),
            "a field only one side has must be reported: {keys:?}"
        );
        assert!(
            !keys
                .iter()
                .any(|k| k.contains("pid") || k.contains("uptime")),
            "per-process noise must be skipped: {keys:?}"
        );
        // The one-sided field shows what is missing rather than inventing a value.
        let kv = found
            .iter()
            .find(|(k, _, _)| k.ends_with("kv_storage"))
            .unwrap();
        assert_eq!((kv.1.as_str(), kv.2.as_str()), ("F16", "—"));
    }

    /// Two identical configurations produce no diff at all — otherwise the
    /// diff is noise and nobody reads the one that matters.
    #[test]
    fn identical_configurations_diff_to_nothing() {
        let records = vec![rec("pp", 1024, 163.3)];
        let props = serde_json::json!({"gpu": {"flags": {"a": true}}, "pid": 7});
        let a = round_trip(props.clone(), &records);
        let b = round_trip(props, &records);
        assert_eq!(diff(&a, &b), Vec::new());
    }
}
