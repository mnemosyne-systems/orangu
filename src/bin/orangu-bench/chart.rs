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

//! Renders a [`history`](super::history) file as a standalone SVG: two charts,
//! prompt processing and token generation, each plotting **tokens/second
//! against context length** with one line per engine.
//!
//! Two charts rather than one panel per workload because the question the file
//! is kept to answer is how throughput behaves *as context grows* — whether a
//! curve is flat or falling away, and how far apart two engines are along it.
//! That is a shape, and a shape needs the workload on an axis, not spread
//! across facets. Prefill and decode stay separate because they are different
//! measurements that happen to share a unit.
//!
//! The y-axis is **logarithmic**. The engines on it differ by an order of
//! magnitude, so on a linear axis the slower one collapses onto the baseline
//! and its own shape — the thing being tracked — becomes unreadable. On a log
//! axis a constant ratio is a constant vertical distance, which is exactly how
//! "N× behind" should read.
//!
//! Only the **newest measurement date** in the file is drawn. The file keeps
//! every run — that is what it is for — but a chart of "how does throughput
//! behave as context grows" is answered by the current state, and overlaying
//! superseded runs on top of it just crowds the lines it is being read for.
//! Reach for the history by reading the file, or by pointing `--chart` at a
//! filtered copy of it.
//!
//! The output is a single self-contained file with no external references, so
//! it renders the same in a browser, in a Markdown preview and in a diff.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::history::Record;

/// Categorical hues, assigned to labels in first-seen order and never cycled:
/// past the last slot a series is dropped from the chart rather than given a
/// colour another series already owns. `(light, dark)` steps of the same hue.
const SERIES_COLORS: [(&str, &str); 6] = [
    ("#2a78d6", "#3987e5"), // blue
    ("#eb6834", "#d95926"), // orange
    ("#1baf7a", "#199e70"), // aqua
    ("#eda100", "#c98500"), // yellow
    ("#e87ba4", "#d55181"), // magenta
    ("#4a3aa7", "#9085e9"), // violet
];

const WIDTH: f64 = 900.0;
const PANEL_H: f64 = 330.0;
const PAD_L: f64 = 64.0;
const PAD_R: f64 = 108.0; // room for the direct labels at each line's right end
const PAD_T: f64 = 34.0;
const PAD_B: f64 = 48.0;

/// One engine's curve on one chart: its rate at each context length, on one
/// measurement date.
struct Line {
    color_idx: usize,
    label: String,
    date: String,
    /// `(context length, tok/s)`, ascending by context.
    points: Vec<(u32, f64)>,
}

/// One of the two charts.
struct Chart {
    title: String,
    x_title: String,
    lines: Vec<Line>,
}

/// Render `records` to an SVG document.
///
/// Repeated measurements of the same series at the same context on the same
/// date collapse to their best, matching how a single run reports its own
/// repetitions: the chart is read for what a build can do, and a slower rerun
/// on the same day is the machine being busy, not the build regressing.
pub fn render(all_records: &[Record], subtitle: &str) -> String {
    // Current state only: the newest date present, per the module docs.
    let newest_date = all_records
        .iter()
        .map(|r| r.date.as_str())
        .max()
        .unwrap_or("");
    let records: Vec<Record> = all_records
        .iter()
        .filter(|r| r.date == newest_date)
        .cloned()
        .collect();
    let records = &records[..];
    let mut labels: Vec<String> = Vec::new();
    for r in records {
        if !labels.iter().any(|l| l == &r.label) {
            labels.push(r.label.clone());
        }
    }
    labels.truncate(SERIES_COLORS.len());

    let mut charts = Vec::new();
    for (mode, title, x_title) in [
        (
            "pp",
            "Prefill — prompt processing",
            "prompt length (tokens)",
        ),
        ("tg", "Decode — token generation", "context length (tokens)"),
    ] {
        // `(label, date) -> context -> best`.
        let mut by_series: BTreeMap<(usize, &str, &str), BTreeMap<u32, f64>> = BTreeMap::new();
        for r in records.iter().filter(|r| r.mode == mode) {
            let Some(ci) = labels.iter().position(|l| l == &r.label) else {
                continue;
            };
            let at = by_series
                .entry((ci, r.label.as_str(), r.date.as_str()))
                .or_default()
                .entry(r.n)
                .or_insert(r.best);
            if r.best > *at {
                *at = r.best;
            }
        }
        if by_series.is_empty() {
            continue;
        }
        let lines = by_series
            .into_iter()
            .map(|((ci, label, date), pts)| Line {
                color_idx: ci,
                label: label.to_string(),
                date: date.to_string(),
                points: pts.into_iter().collect(),
            })
            .collect();
        charts.push(Chart {
            title: title.to_string(),
            x_title: x_title.to_string(),
            lines,
        });
    }

    let head_h = 56.0;
    let legend_h = 30.0;
    let height = head_h + legend_h + PANEL_H * charts.len() as f64 + 10.0;

    let mut s = String::new();
    let _ = write!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH:.0} {height:.0}" width="{WIDTH:.0}" height="{height:.0}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">
<style>
  :root {{ color-scheme: light dark; }}
  .surface {{ fill: #fcfcfb; }}
  .ink {{ fill: #0b0b0b; }}
  .ink-2 {{ fill: #52514e; }}
  .grid {{ stroke: #dcdbd6; stroke-width: 1; }}
  .axis {{ stroke: #b4b3ac; stroke-width: 1; }}
"##
    );
    // Lines carry a series colour as `stroke` only. A CSS rule beats a
    // presentation attribute, so a class that also set `fill` would override
    // the `fill="none"` on every polyline and paint each curve as a filled
    // area under itself — which is exactly what it did. The dot fills are a
    // separate set of classes, emitted after, and only markers wear them.
    for (i, color) in SERIES_COLORS.iter().enumerate().take(labels.len()) {
        let _ = writeln!(s, "  .s{i} {{ stroke: {}; }}", color.0);
    }
    for (i, color) in SERIES_COLORS.iter().enumerate().take(labels.len()) {
        let _ = writeln!(s, "  .f{i} {{ fill: {}; }}", color.0);
    }
    // Last, so the later rule wins and an overlapping marker keeps its
    // surface-coloured separating ring.
    let _ = writeln!(s, "  .mark-ring {{ stroke: #fcfcfb; stroke-width: 2; }}");
    // Dark mode is selected, not flipped: its own step of each hue against the
    // dark surface. Declared under both the OS media query and the explicit
    // theme attribute so a viewer's toggle wins in either direction.
    let mut dark = String::new();
    let _ = write!(
        dark,
        "    .surface {{ fill: #1a1a19; }}\n    .ink {{ fill: #ffffff; }}\n    .ink-2 {{ fill: #c3c2b7; }}\n    .grid {{ stroke: #35342f; }}\n    .axis {{ stroke: #56554e; }}\n"
    );
    for (i, color) in SERIES_COLORS.iter().enumerate().take(labels.len()) {
        let _ = writeln!(
            dark,
            "    .s{i} {{ stroke: {}; fill: {}; }}",
            color.1, color.1
        );
    }
    let _ = writeln!(
        dark,
        "    .mark-ring {{ stroke: #1a1a19; stroke-width: 2; }}"
    );
    let _ = write!(
        s,
        "  @media (prefers-color-scheme: dark) {{\n{dark}  }}\n  :root[data-theme=\"dark\"] {{\n{dark}  }}\n</style>\n"
    );

    let _ = write!(
        s,
        r#"<rect class="surface" x="0" y="0" width="{WIDTH:.0}" height="{height:.0}"/>
<text class="ink" x="16" y="26" font-size="16" font-weight="600">orangu throughput vs context</text>
<text class="ink-2" x="16" y="44" font-size="11">{} · showing {} ({} of {} rows)</text>
"#,
        esc(subtitle),
        esc(newest_date),
        records.len(),
        all_records.len()
    );

    // Identity is never colour alone: every series is named here and each
    // line's right-hand end is direct-labelled in the chart below.
    let mut lx = 16.0;
    for (i, label) in labels.iter().enumerate() {
        let _ = write!(
            s,
            r#"<circle class="f{i}" cx="{:.1}" cy="{:.1}" r="4"/><text class="ink" x="{:.1}" y="{:.1}" font-size="11">{}</text>"#,
            lx + 4.0,
            head_h + 6.0,
            lx + 14.0,
            head_h + 10.0,
            esc(label)
        );
        s.push('\n');
        lx += 22.0 + 6.6 * label.chars().count() as f64;
    }

    for (idx, chart) in charts.iter().enumerate() {
        render_chart(&mut s, chart, head_h + legend_h + idx as f64 * PANEL_H);
    }

    s.push_str("</svg>\n");
    s
}

fn render_chart(s: &mut String, chart: &Chart, oy: f64) {
    let plot_w = WIDTH - PAD_L - PAD_R;
    let plot_h = PANEL_H - PAD_T - PAD_B;
    let x0 = PAD_L;
    let y0 = oy + PAD_T;

    let all: Vec<(u32, f64)> = chart
        .lines
        .iter()
        .flat_map(|l| l.points.iter().copied())
        .collect();
    let x_max = all.iter().map(|(n, _)| *n).max().unwrap_or(1).max(1) as f64;
    let v_min = all.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
    let v_max = all.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max);
    // Decade-aligned bounds, so every gridline is a round number and the ratio
    // between two curves can be read off the axis.
    let lo = axis_floor(v_min.max(1e-6));
    let hi = axis_ceil(v_max.max(lo * 1.5));

    // Context starts at 0 (a decode depth really is 0), so the x-axis is linear
    // from zero rather than from the smallest measured length.
    let px = |n: u32| -> f64 { x0 + plot_w * (n as f64 / x_max) };
    let py = |v: f64| -> f64 {
        let t = (v.max(1e-6).log10() - lo.log10()) / (hi.log10() - lo.log10());
        y0 + plot_h - plot_h * t.clamp(0.0, 1.0)
    };

    let _ = write!(
        s,
        r#"<text class="ink" x="{x0:.1}" y="{:.1}" font-size="13" font-weight="600">{}</text>"#,
        y0 - 14.0,
        esc(&chart.title)
    );
    s.push('\n');

    // Log gridlines at 1/2/5 × each decade.
    let mut decade = 10f64.powf(lo.log10().floor());
    while decade <= hi + 1e-9 {
        for m in [1.0, 2.0, 5.0] {
            let v = decade * m;
            if v < lo - 1e-9 || v > hi + 1e-9 {
                continue;
            }
            let y = py(v);
            let _ = write!(
                s,
                r#"<line class="grid" x1="{x0:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}"/><text class="ink-2" x="{:.1}" y="{:.1}" font-size="9" text-anchor="end">{}</text>"#,
                x0 + plot_w,
                x0 - 7.0,
                y + 3.0,
                fmt_num(v)
            );
            s.push('\n');
        }
        decade *= 10.0;
    }
    let _ = write!(
        s,
        r#"<line class="axis" x1="{x0:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}"/>"#,
        y0 + plot_h,
        x0 + plot_w,
        y0 + plot_h
    );
    s.push('\n');
    let _ = write!(
        s,
        r#"<text class="ink-2" x="{:.1}" y="{:.1}" font-size="10" transform="rotate(-90 {:.1} {:.1})" text-anchor="middle">tok/s (log)</text>"#,
        x0 - 46.0,
        y0 + plot_h / 2.0,
        x0 - 46.0,
        y0 + plot_h / 2.0
    );
    s.push('\n');

    // X ticks at the context lengths actually measured — they are the only
    // places the curves carry information.
    let mut xs: Vec<u32> = all.iter().map(|(n, _)| *n).collect();
    xs.sort_unstable();
    xs.dedup();
    let mut last_x = f64::NEG_INFINITY;
    for n in xs {
        let x = px(n);
        if x - last_x < 34.0 {
            continue;
        }
        last_x = x;
        let _ = write!(
            s,
            r#"<line class="grid" x1="{x:.1}" y1="{:.1}" x2="{x:.1}" y2="{:.1}"/><text class="ink-2" x="{x:.1}" y="{:.1}" font-size="9" text-anchor="middle">{n}</text>"#,
            y0,
            y0 + plot_h,
            y0 + plot_h + 15.0
        );
        s.push('\n');
    }
    let _ = write!(
        s,
        r#"<text class="ink-2" x="{:.1}" y="{:.1}" font-size="10" text-anchor="middle">{}</text>"#,
        x0 + plot_w / 2.0,
        y0 + plot_h + 34.0,
        esc(&chart.x_title)
    );
    s.push('\n');

    // Two passes: every line, then every dot. Interleaving them lets one
    // series' line paint over another's markers, which is what made curves
    // this close together read as one smeared band. Lines are the mark that
    // carries the shape; the dots only say where a measurement actually
    // exists, so they belong on top of all of it.
    let order: Vec<&Line> = chart.lines.iter().collect();

    for line in &order {
        if line.points.len() < 2 {
            continue;
        }
        let ci = line.color_idx;
        let d: Vec<String> = line
            .points
            .iter()
            .map(|(n, v)| format!("{:.1},{:.1}", px(*n), py(*v)))
            .collect();
        let _ = write!(
            s,
            r#"<polyline class="s{ci}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="{}"/>"#,
            d.join(" ")
        );
        s.push('\n');
    }

    for line in &order {
        let ci = line.color_idx;
        for (n, v) in &line.points {
            let _ = write!(
                s,
                r#"<circle class="f{ci} mark-ring" cx="{:.1}" cy="{:.1}" r="4"><title>{} · {} · {n} tok · {v:.2} tok/s</title></circle>"#,
                px(*n),
                py(*v),
                esc(&line.label),
                esc(&line.date)
            );
            s.push('\n');
        }
    }

    // Direct-label each line's right-hand end — the number a reader wants
    // without matching a colour to the legend. Nudged vertically off any label
    // already placed, since two engines can end close together.
    let mut placed: Vec<f64> = Vec::new();
    for line in &order {
        if let Some((n, v)) = line.points.last() {
            let mut y = py(*v) + 3.5;
            while placed.iter().any(|oy| (oy - y).abs() < 11.0) {
                y += 11.0;
            }
            placed.push(y);
            let _ = write!(
                s,
                r#"<text class="ink" x="{:.1}" y="{y:.1}" font-size="10" font-weight="600">{}</text>"#,
                px(*n) + 9.0,
                fmt_num(*v)
            );
            s.push('\n');
        }
    }
}

/// Largest `1/2/5 × 10^k` at or below `v` — a whole decade of headroom below
/// the slowest measurement would leave every curve squashed into the top third.
fn axis_floor(v: f64) -> f64 {
    let d = 10f64.powf(v.log10().floor());
    for m in [5.0, 2.0] {
        if v >= d * m {
            return d * m;
        }
    }
    d
}

/// Smallest `1/2/5 × 10^k` at or above `v`.
fn axis_ceil(v: f64) -> f64 {
    let d = 10f64.powf(v.log10().floor());
    for m in [1.0, 2.0, 5.0] {
        if v <= d * m {
            return d * m;
        }
    }
    d * 10.0
}

/// Axis and direct-label numbers: whole once the value is big enough for a
/// decimal to be noise, one decimal below that so a log axis's low end does not
/// print `2 2 5` for 2.0/2.5/5.0.
fn fmt_num(v: f64) -> String {
    if v >= 10.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}

/// XML-escape text destined for a text node or an attribute value.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(date: &str, label: &str, mode: &str, n: u32, best: f64) -> Record {
        Record {
            date: date.into(),
            label: label.into(),
            mode: mode.into(),
            n,
            best,
            mean: best,
            sd: 0.0,
        }
    }

    #[test]
    fn there_are_exactly_two_charts_prefill_and_decode() {
        let svg = render(
            &[
                rec("2026-07-25", "orangu", "pp", 158, 89.0),
                rec("2026-07-25", "orangu", "pp", 1120, 112.0),
                rec("2026-07-25", "orangu", "tg", 0, 43.0),
                rec("2026-07-25", "orangu", "tg", 1024, 29.0),
            ],
            "test",
        );
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>\n"));
        assert_eq!(svg.matches("Prefill — prompt processing").count(), 1);
        assert_eq!(svg.matches("Decode — token generation").count(), 1);
        assert!(svg.contains("prompt length (tokens)"));
        assert!(svg.contains("context length (tokens)"));
    }

    #[test]
    fn a_series_across_contexts_is_one_line_not_scattered_points() {
        let svg = render(
            &[
                rec("2026-07-25", "orangu", "pp", 158, 89.0),
                rec("2026-07-25", "orangu", "pp", 574, 114.0),
                rec("2026-07-25", "orangu", "pp", 1120, 112.0),
            ],
            "t",
        );
        // One polyline through all three contexts.
        assert_eq!(svg.matches("<polyline").count(), 1);
        assert_eq!(svg.matches("<circle").count(), 3 + 1); // + the legend swatch
    }

    #[test]
    fn each_engine_gets_its_own_line_in_its_own_colour() {
        let svg = render(
            &[
                rec("2026-07-25", "orangu", "pp", 158, 89.0),
                rec("2026-07-25", "orangu", "pp", 1120, 112.0),
                rec("2026-07-25", "llama.cpp", "pp", 158, 818.0),
                rec("2026-07-25", "llama.cpp", "pp", 1120, 1062.0),
            ],
            "t",
        );
        assert_eq!(svg.matches("<polyline").count(), 2);
        assert!(svg.contains("#2a78d6") && svg.contains("#eb6834"));
        // Lines must not be filled: a class that set `fill` would beat the
        // `fill="none"` attribute and paint each curve as a filled area.
        assert!(!svg.contains(".s0 { stroke: #2a78d6; fill:"));
        assert!(svg.contains(".s0 { stroke: #2a78d6; }"));
        assert!(svg.contains(".f0 { fill: #2a78d6; }"));
        for poly in svg.split("<polyline").skip(1) {
            let tag = &poly[..poly.find("/>").unwrap()];
            assert!(
                tag.contains(r#"fill="none""#),
                "unfilled line expected: {tag}"
            );
            assert!(!tag.contains(".f"), "a line must not wear a fill class");
        }
        assert!(svg.contains(">orangu<") && svg.contains(">llama.cpp<"));
    }

    /// The file accumulates every run; the chart is only ever the newest date
    /// in it. A superseded run must leave no line, no marker and no direct
    /// label — otherwise the chart is read as a comparison between two builds
    /// when one of them no longer exists.
    #[test]
    fn only_the_newest_date_is_drawn() {
        let svg = render(
            &[
                rec("2026-07-25", "orangu", "pp", 158, 60.0),
                rec("2026-07-25", "orangu", "pp", 1120, 80.0),
                rec("2026-07-25", "orangu pre-P1", "pp", 158, 40.0),
                rec("2026-07-25", "orangu pre-P1", "pp", 1120, 50.0),
                rec("2026-08-01", "orangu", "pp", 158, 89.0),
                rec("2026-08-01", "orangu", "pp", 1120, 112.0),
            ],
            "t",
        );
        // One series, one line — not three.
        assert_eq!(svg.matches("<polyline").count(), 1);
        assert_eq!(svg.matches(">112<").count(), 1);
        assert_eq!(svg.matches(">80<").count(), 0);
        // A label that only appears on the superseded date is gone entirely,
        // legend included.
        assert!(!svg.contains("pre-P1"));
        assert_eq!(svg.matches(">orangu<").count(), 1);
        // No fading left over from when history was overlaid.
        assert!(!svg.contains("stroke-opacity"));
    }

    #[test]
    fn the_log_axis_covers_both_engines_with_round_gridlines() {
        let svg = render(
            &[
                rec("2026-07-25", "orangu", "pp", 1120, 112.0),
                rec("2026-07-25", "llama.cpp", "pp", 1120, 1062.0),
            ],
            "t",
        );
        assert!(svg.contains("tok/s (log)"));
        // 1/2/5 bounds hugging 112..1062, not a whole decade either side.
        assert_eq!(axis_floor(112.0), 100.0);
        assert_eq!(axis_ceil(1062.0), 2000.0);
        assert_eq!(axis_floor(25.63), 20.0);
        assert_eq!(axis_ceil(43.0), 50.0);
        assert!(svg.contains(">100<") && svg.contains(">1000<"));
    }

    #[test]
    fn markup_from_a_label_is_escaped_not_emitted() {
        let svg = render(&[rec("2026-07-25", "a<b>&\"c", "pp", 8, 1.0)], "t");
        assert!(!svg.contains("a<b>"));
        assert!(svg.contains("a&lt;b&gt;&amp;&quot;c"));
    }

    #[test]
    fn labels_past_the_last_colour_slot_are_dropped_not_recoloured() {
        let mut recs = Vec::new();
        for i in 0..SERIES_COLORS.len() + 2 {
            recs.push(rec("2026-07-25", &format!("build{i}"), "pp", 1024, 10.0));
        }
        let svg = render(&recs, "t");
        assert!(!svg.contains(&format!(">build{}<", SERIES_COLORS.len() + 1)));
        assert!(svg.contains(">build0<"));
    }
}
