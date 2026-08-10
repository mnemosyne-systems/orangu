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

//! The append-only throughput history behind `--history`, and the record type
//! [`chart`](super::chart) renders.
//!
//! A benchmark run is only interesting next to another one, and the two things
//! a rate must be compared against — *the other engine* and *last week's
//! build* — are both outside any single invocation. So each measured point is
//! appended to a tab-separated file that accumulates across runs, and the chart
//! is drawn from the file rather than from the run that happened to produce it.
//!
//! Tab-separated rather than JSON on purpose: the file is meant to be read in a
//! terminal, diffed in a review, and committed next to the code it describes.
//!
//! Nothing is ever rewritten — a row is a measurement that was taken, and a
//! later run that disagrees is another row, not a correction.

use std::fmt::Write as _;
use std::io::Write as _;

/// The column header written when a history file is created, so a file found
/// on its own is self-describing. Read back as a comment and skipped.
const HEADER: &str = "#date\tlabel\tmode\tn\tbest\tmean\tsd\tsd_sample\tdevice";

/// One measurement: what was measured, of what, when, and how fast it went.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    /// `YYYY-MM-DD` in UTC — the resolution a trend is read at.
    pub date: String,
    /// Which engine/build produced it (`--label`). The series identity in the
    /// chart, so it must stay stable across runs to draw a line.
    pub label: String,
    /// `pp` (prompt processing), `tg` (token generation), `curve` (decode rate
    /// bucketed by context, from one pass), `cpu` (CPU ms per token) or
    /// `embed`. Each is drawn as its own chart panel, because they are
    /// different measurements — two of them are not even in tokens/second.
    pub mode: String,
    /// Prompt length for `pp`/`embed`, context depth for `tg`/`cpu`, the
    /// bucket's starting context for `curve`.
    pub n: u32,
    /// Best of the run's repetitions, in tokens/second — the same statistic
    /// the table prints, chosen for the same reason: it is the one least
    /// contaminated by an unrelated process getting scheduled mid-run.
    pub best: f64,
    pub mean: f64,
    /// Population standard deviation (÷ n) — what this column has meant since
    /// the file was created, and therefore what it means for every row in it.
    /// A column in an append-only record is not redefined; a new one is added.
    pub sd: f64,
    /// Sample standard deviation (÷ n-1), the standard estimator — the figure
    /// that can be put beside a `±` from another benchmark. Written as an
    /// eighth column, empty where a run of one repetition leaves it undefined
    /// and absent from every row written before it existed.
    pub sd_sample: Option<f64>,
    /// Which device produced the row — the server's own backend label, short
    /// enough to sit in a legend (see `orangu-bench`'s `device_tag`).
    ///
    /// A ninth column rather than something folded into [`Record::label`],
    /// because a row's device is a fact about the run and a label is a name
    /// somebody chose: with `--device` on the server, the same model measured
    /// on two cards produced two sets of rows that were identical in every
    /// recorded field, and the only thing standing between them and a chart
    /// drawn as one series was the operator having remembered `--label`.
    /// A recorded fact belongs in a column of its own, not folded into a
    /// name somebody has to remember to set.
    ///
    /// `None` on every row written before this column existed, and on a server
    /// that reports no backend at all. Not defaulted to anything: "unknown
    /// device" and "some particular device" are different claims, and only one
    /// of them can be compared against a later run.
    pub device: Option<String>,
}

impl Record {
    fn to_row(&self) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}",
            self.date,
            self.label,
            self.mode,
            self.n,
            self.best,
            self.mean,
            self.sd,
            // Empty rather than `0.00`: one repetition has no sample spread,
            // and a zero would claim it was measured.
            self.sd_sample
                .map_or_else(String::new, |sd| format!("{sd:.2}")),
            self.device.as_deref().unwrap_or(""),
        );
        s
    }

    fn from_row(line: &str) -> Option<Self> {
        let mut f = line.split('\t');
        let rec = Record {
            date: f.next()?.to_string(),
            label: f.next()?.to_string(),
            mode: f.next()?.to_string(),
            n: f.next()?.trim().parse().ok()?,
            best: f.next()?.trim().parse().ok()?,
            mean: f.next()?.trim().parse().ok()?,
            sd: f.next()?.trim().parse().ok()?,
            // Absent on every row written before this column existed, and
            // empty on a single-repetition row. Neither is a malformed row:
            // the file is append-only, so old rows stay exactly as they were.
            sd_sample: f.next().and_then(|v| v.trim().parse().ok()),
            // Same contract, one column further along: absent on an older row,
            // empty when the server named no backend.
            device: f
                .next()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_string),
        };
        Some(rec)
    }

    /// The series this row belongs to in a chart: the label alone, or the
    /// label and the device when `show_device` says the set it came from holds
    /// more than one.
    ///
    /// `?` for a row that records no device, which is what an unqualified
    /// label has always meant — it is only worth spelling out once a
    /// neighbouring row *does* name one.
    pub fn series(&self, show_device: bool) -> String {
        if show_device {
            format!("{} · {}", self.label, self.device.as_deref().unwrap_or("?"))
        } else {
            self.label.clone()
        }
    }
}

/// Whether `records` hold measurements from more than one named device — the
/// condition under which a label alone no longer identifies a series.
///
/// Rows that name **no** device are ignored rather than counted as a device of
/// their own. Every row written before the column existed is one of those, so
/// counting them would rename every historical series the moment one new run
/// appended a row beside them — a file whose old rows did not change would
/// draw a chart whose old lines did. The rule is the one a table follows when
/// it grows a column only for a parameter that varies.
pub fn devices_differ(records: &[Record]) -> bool {
    let mut seen: Option<&str> = None;
    for device in records.iter().filter_map(|r| r.device.as_deref()) {
        match seen {
            None => seen = Some(device),
            Some(first) if first != device => return true,
            Some(_) => {}
        }
    }
    false
}

/// Append `records` to `path`, creating it with a header if it does not exist.
pub fn append(path: &str, records: &[Record]) -> anyhow::Result<()> {
    let fresh = !std::path::Path::new(path).exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if fresh {
        writeln!(file, "{HEADER}")?;
    }
    for r in records {
        writeln!(file, "{}", r.to_row())?;
    }
    Ok(())
}

/// Read every record from `path`. Blank lines and `#` comments are skipped, and
/// so is any row that does not parse — a history file is hand-editable, and one
/// malformed line should not cost the chart every good row around it.
pub fn read(path: &str) -> anyhow::Result<Vec<Record>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(Record::from_row)
        .collect())
}

/// Today's date in UTC as `YYYY-MM-DD`.
///
/// Done from `SystemTime` and the civil-from-days algorithm rather than by
/// taking a date-time dependency: this is the only place in the tool that needs
/// a calendar, and it needs one field of it.
pub fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The moment a run was measured, as `2026-08-04T09:34:21Z`.
///
/// A date alone cannot separate two runs on the same afternoon, which is
/// exactly when an A/B is taken — so the *bundle* records this while the
/// history file keeps its date column. UTC and RFC 3339 so a bundle carried to
/// another machine, in another zone, still sorts and still means one instant.
pub fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let day = secs % 86_400;
    let (hh, mm, ss) = (day / 3600, (day % 3600) / 60, day % 60);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Days since 1970-01-01 to a `(year, month, day)` civil date (proleptic
/// Gregorian). Howard Hinnant's `civil_from_days`, which shifts the epoch to
/// 0000-03-01 so leap days land at the end of the era's 400-year cycle and the
/// month arithmetic becomes branch-free.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March-based
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The timestamp has to be the same instant as the date, and shaped so it
    /// sorts as text — a bundle's whole job is to be read back later, often
    /// beside another one.
    #[test]
    fn the_timestamp_agrees_with_the_date_and_sorts() {
        let stamp = now_utc();
        assert!(stamp.starts_with(&today()), "{stamp} vs {}", today());
        assert!(stamp.ends_with('Z') && stamp.len() == 20, "{stamp}");
        assert!("2026-08-04T09:34:21Z" < "2026-08-04T09:34:22Z");
        assert!("2026-08-04T23:59:59Z" < "2026-08-05T00:00:00Z");
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        // 2000-02-29: a leap day in a century that *is* a leap year, the case
        // the 400-year era term exists for.
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(civil_from_days(11_017), (2000, 3, 1));
        // 1900 was not a leap year; 1900-02-28 is followed by 1900-03-01.
        assert_eq!(civil_from_days(-25_509), (1900, 2, 28));
        assert_eq!(civil_from_days(-25_508), (1900, 3, 1));
        assert_eq!(civil_from_days(20_293), (2025, 7, 24));
    }

    #[test]
    fn a_row_survives_a_round_trip() {
        let r = Record {
            date: "2026-07-25".into(),
            label: "orangu".into(),
            mode: "pp".into(),
            n: 1024,
            best: 81.75,
            mean: 81.40,
            sd_sample: Some(0.31),
            sd: 0.26,
            device: Some("api/first card".into()),
        };
        assert_eq!(Record::from_row(&r.to_row()), Some(r));
    }

    /// The device column arrived the same way `sd_sample` did — appended to a
    /// file that already had years of rows in it — and has to behave the same
    /// way: an eight-column row is a measurement whose device nobody recorded,
    /// not a broken line.
    ///
    /// `perf-history.tsv` currently holds seven-, eight- and nine-column rows
    /// at once. All three have to read.
    #[test]
    fn a_row_written_before_the_device_column_still_reads() {
        let eight = "2026-07-25\torangu\tpp\t1120\t81.75\t81.40\t0.26\t0.31";
        let parsed = Record::from_row(eight).expect("an eight-column row still parses");
        assert_eq!(parsed.sd_sample, Some(0.31));
        assert_eq!(parsed.device, None, "a device was invented from nothing");

        // A nine-column row whose device is empty means the same thing: the
        // server named no backend. Empty is not a device called "".
        let empty = "2026-07-25\torangu\tpp\t1120\t81.75\t81.40\t0.26\t0.31\t";
        assert_eq!(Record::from_row(empty).unwrap().device, None);
    }

    /// The reason this column exists. Two runs of one model on two cards
    /// differ in *no other recorded field* — same date, same label, same mode,
    /// same n — so without the device they are one series with two points at
    /// every x, and a chart of them is a comparison silently drawn as a trend.
    #[test]
    fn one_model_on_two_cards_is_two_series() {
        let on = |device: &str, best: f64| Record {
            date: "2026-08-10".into(),
            label: "gemma-4-E2B".into(),
            mode: "tg".into(),
            n: 0,
            best,
            mean: best,
            sd: 0.0,
            sd_sample: None,
            device: Some(device.into()),
        };
        let rows = [on("api/first card", 43.0), on("api/second card", 21.5)];
        assert!(devices_differ(&rows));
        assert_ne!(rows[0].series(true), rows[1].series(true));
        // And the whole point of the flag: with the device dropped they are
        // indistinguishable, which is what every row in this file looked like
        // before the column existed.
        assert_eq!(rows[0].series(false), rows[1].series(false));
    }

    /// One card is the ordinary case, and it must not rename anything: a
    /// legend that reads `orangu · api/first card` on a machine with one GPU
    /// is noise, and worse, it breaks the series identity of a file whose
    /// earlier rows say `orangu`.
    #[test]
    fn one_device_and_unrecorded_devices_leave_the_series_alone() {
        let plain = Record {
            date: "2026-08-10".into(),
            label: "orangu".into(),
            mode: "tg".into(),
            n: 0,
            best: 43.0,
            mean: 43.0,
            sd: 0.0,
            sd_sample: None,
            device: None,
        };
        let named = Record {
            device: Some("api/first card".into()),
            ..plain.clone()
        };
        // Two rows on the same card.
        assert!(!devices_differ(&[named.clone(), named.clone()]));
        // Historical rows that name no device are not a second device: a file
        // of old rows plus one new run on one card is still one card.
        assert!(!devices_differ(&[plain.clone(), named.clone()]));
        assert!(!devices_differ(&[plain.clone(), plain.clone()]));
        assert_eq!(named.series(false), "orangu");
    }

    /// Every row written before `sd_sample` existed has seven columns, and
    /// this file is append-only: those rows are the record of what was
    /// measured and are never rewritten. So a short row must read back as
    /// "not recorded" — not as a parse failure that silently drops years of
    /// history from a chart.
    #[test]
    fn a_row_written_before_the_sample_column_still_reads() {
        let old = "2026-07-25\torangu af7c767\tpp\t1120\t81.75\t81.40\t0.26";
        let parsed = Record::from_row(old).expect("an older row still parses");
        assert_eq!(parsed.n, 1120);
        assert!((parsed.sd - 0.26).abs() < 1e-9);
        assert_eq!(parsed.sd_sample, None);

        // And a single-repetition row written by *this* build leaves the
        // column empty rather than claiming a zero spread.
        let one_rep = Record {
            sd_sample: None,
            ..parsed.clone()
        };
        assert!(one_rep.to_row().ends_with('\t'), "{:?}", one_rep.to_row());
        assert_eq!(Record::from_row(&one_rep.to_row()), Some(one_rep));
    }

    #[test]
    fn comments_blanks_and_junk_are_skipped_not_fatal() {
        let dir = std::env::temp_dir().join(format!("orangu-bench-hist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.tsv");
        let p = path.to_str().unwrap();
        std::fs::write(
            &path,
            "#date\tlabel\tmode\tn\tbest\tmean\tsd\n\
             \n\
             2026-07-25\torangu\tpp\t1024\t81.75\t81.40\t0.26\n\
             this is not a row\n\
             2026-07-25\treference\tpp\t1024\t1061.66\t1049.40\t8.84\n",
        )
        .unwrap();
        let recs = read(p).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].label, "orangu");
        assert_eq!(recs[1].best, 1061.66);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_creates_the_header_once_and_only_once() {
        let dir = std::env::temp_dir().join(format!("orangu-bench-app-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.tsv");
        let p = path.to_str().unwrap();
        let r = Record {
            date: "2026-07-25".into(),
            label: "orangu".into(),
            mode: "tg".into(),
            n: 0,
            best: 43.07,
            mean: 42.75,
            sd: 0.26,
            sd_sample: Some(0.31),
            device: None,
        };
        append(p, std::slice::from_ref(&r)).unwrap();
        append(p, std::slice::from_ref(&r)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches(HEADER).count(), 1);
        assert_eq!(read(p).unwrap().len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
