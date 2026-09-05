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

//! Collapsing `perf script` output into folded stacks, and rendering those to
//! an SVG flamegraph — in-process, with no external scripts.
//!
//! This replaces `stackcollapse-perf.pl`, `stackcollapse-recursive.pl` and
//! `flamegraph.pl`. The only external dependency left on the profiling path is
//! `perf` itself, which is the one piece that genuinely cannot be
//! reimplemented: it reads the kernel's perf events.
//!
//! Dropping the Perl was not only about dependency count. The three scripts
//! have to be fetched as a separate checkout, are found by a search path that
//! can silently pick a stale copy, and — most concretely — the recursion
//! folding they offer does not cover the shape this project actually produces.
//! `stackcollapse-recursive.pl` folds a frame repeated *adjacently*; rayon's
//! splitter alternates several frames per level, so nothing adjacent repeats,
//! every split depth owns a distinct stack of hundred-character monomorphized
//! names, and one 26-second prefill profile collapsed to a **655 MB** text
//! file. [`collapse`] folds whole cycles instead, which takes the same profile
//! to 6 MB with every sample retained.
//!
//! The rendered SVG keeps what the flamegraphs in this project are read for:
//! click a frame to zoom into it, click the header to reset, and Ctrl-F to
//! highlight everything matching a substring with the matched fraction
//! reported. It is one self-contained file with no external references.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The suffix put on kernel-mode frames — the same mark
/// `stackcollapse-perf.pl --kernel` used, so collapsed files from before this
/// module and after it classify identically.
pub const KERNEL_MARK: &str = "_[k]";

/// Collapse `perf script` output into `stack count` lines.
///
/// `perf script`'s default call-graph output is a sample header line followed
/// by indented frames, **leaf first**, terminated by a blank line:
///
/// ```text
/// orangu-server 1234 [003] 98.7654:  250000 cycles:ppp:
///         7f0a1234abcd dot_avx2+0x1c (/path/to/orangu-server)
///         7f0a1234ab00 run_layers+0x8 (/path/to/orangu-server)
///     ffffffff81234567 do_syscall_64+0x59 ([kernel.kallsyms])
/// ```
///
/// Frames come out root-first with the thread name as the root, symbol offsets
/// stripped, kernel frames marked, and recursion cycles folded.
pub fn collapse(perf_script: &str) -> BTreeMap<String, u64> {
    let mut totals: BTreeMap<String, u64> = BTreeMap::new();
    let mut comm = String::new();
    let mut frames: Vec<String> = Vec::new();

    let mut flush = |comm: &mut String, frames: &mut Vec<String>| {
        if comm.is_empty() {
            frames.clear();
            return;
        }
        // `perf` prints leaf first; a flamegraph reads root first.
        frames.reverse();
        let mut stack = vec![std::mem::take(comm)];
        stack.append(frames);
        *totals.entry(fold_cycles(&stack)).or_default() += 1;
    };

    for line in perf_script.lines() {
        if line.trim().is_empty() {
            flush(&mut comm, &mut frames);
            continue;
        }
        // A sample header starts in column zero; a frame is indented.
        if !line.starts_with([' ', '\t']) {
            // Two samples with no blank line between them would otherwise merge.
            flush(&mut comm, &mut frames);
            comm = sample_comm(line).unwrap_or_default();
            continue;
        }
        if !comm.is_empty()
            && let Some(frame) = parse_frame(line)
        {
            frames.push(frame);
        }
    }
    flush(&mut comm, &mut frames);
    totals
}

/// The thread name from a `perf script` sample header.
///
/// The header is `comm pid/tid [cpu] time: period event:` and **the comm may
/// contain spaces** (`Chrome_ChildIOT`, but also `tokio-rt worker` on some
/// runtimes), so it cannot be taken as the first whitespace-delimited field.
/// It ends at the last field that is not part of the fixed tail, which is found
/// by walking back from the first field that looks like a pid.
fn sample_comm(header: &str) -> Option<String> {
    let fields: Vec<&str> = header.split_whitespace().collect();
    // Find the pid/tid field: all digits, or `pid/tid`.
    let pid_at = fields.iter().position(|f| {
        let core = f.split_once('/').map_or(*f, |(a, _)| a);
        !core.is_empty() && core.bytes().all(|b| b.is_ascii_digit())
    })?;
    if pid_at == 0 {
        return None;
    }
    Some(fields[..pid_at].join(" "))
}

/// One frame line: `ADDRESS symbol+0xOFFSET (dso)`.
///
/// Returns the symbol with its offset stripped and, for a kernel dso, the
/// kernel mark appended. Kernel-ness comes from the **dso** rather than from
/// the symbol's name, which is the whole reason to read it here: `read_hpet`
/// and `perf_event_update_userpage` are ordinary-looking symbols that live in
/// the kernel, and a name-based guess files them as application code.
fn parse_frame(line: &str) -> Option<String> {
    let line = line.trim();
    // Drop the leading instruction address.
    let rest = line.split_once(' ').map_or(line, |(_, r)| r).trim();
    if rest.is_empty() {
        return None;
    }

    // The dso is the parenthesised tail; a symbol may itself contain
    // parentheses (`operator()`), so take the *last* one that closes the line.
    let (symbol, dso) = match rest.rfind(" (") {
        Some(at) if rest.ends_with(')') => (&rest[..at], &rest[at + 2..rest.len() - 1]),
        _ => (rest, ""),
    };

    let symbol = symbol.trim();
    // `symbol+0x1c` → `symbol`. Only a trailing `+0x…` counts: Rust and C++
    // symbols contain `+` inside generic parameters.
    let symbol = match symbol.rfind("+0x") {
        Some(at) if symbol[at + 3..].bytes().all(|b| b.is_ascii_hexdigit()) => &symbol[..at],
        _ => symbol,
    };
    if symbol.is_empty() {
        return None;
    }

    let mut name = symbol.replace(';', ":"); // `;` is the folded-format separator
    if is_kernel_dso(dso) {
        name.push_str(KERNEL_MARK);
    }
    Some(name)
}

/// Whether a `perf script` dso field names kernel space.
fn is_kernel_dso(dso: &str) -> bool {
    dso.starts_with("[kernel")
        || dso.starts_with("[vdso")
        || dso.starts_with("vmlinux")
        || dso.ends_with(".ko")
        || dso.ends_with(".ko.zst")
}

/// Collapse a stack back to the first occurrence of any repeated frame, and
/// join it with `;`.
///
/// `stackcollapse-recursive.pl` folds a frame repeated *adjacently*
/// (`a;b;b;c` → `a;b;c`). Rayon's splitter does not recurse that way: it
/// alternates through several frames per level (`helper;join;helper;join;…`),
/// so nothing adjacent repeats and nothing folds — and because these are
/// monomorphized Rust closures, every split depth owns a distinct stack of
/// hundred-character names.
///
/// The trade is the same one adjacent-folding already makes: `a;b;c;b;d`
/// becomes `a;b;d`, so a genuinely re-entrant path loses the frames between the
/// two visits. For a parallel splitter — which is what produces these stacks —
/// that is the reading you want: time under the recursion, not time at depth 14
/// of it.
fn fold_cycles(frames: &[String]) -> String {
    let mut kept: Vec<&str> = Vec::with_capacity(frames.len());
    for frame in frames {
        match kept.iter().position(|f| *f == frame.as_str()) {
            Some(at) => kept.truncate(at + 1),
            None => kept.push(frame),
        }
    }
    kept.join(";")
}

/// Serialize collapsed stacks to the folded-stack text format.
pub fn to_folded(totals: &BTreeMap<String, u64>) -> String {
    let mut out = String::new();
    for (stack, count) in totals {
        let _ = writeln!(out, "{stack} {count}");
    }
    out
}

/// One node of the flamegraph tree.
struct Node {
    name: String,
    total: u64,
    children: Vec<Node>,
}

impl Node {
    fn child(&mut self, name: &str) -> &mut Node {
        // Linear scan: a frame's fan-out is small, and this keeps children in
        // first-seen order so a re-render of the same input is byte-identical.
        if let Some(at) = self.children.iter().position(|c| c.name == name) {
            return &mut self.children[at];
        }
        self.children.push(Node {
            name: name.to_string(),
            total: 0,
            children: Vec::new(),
        });
        self.children.last_mut().expect("just pushed")
    }

    /// Depth-first in name order, so the rendered SVG is deterministic.
    fn sort(&mut self) {
        self.children.sort_by(|a, b| a.name.cmp(&b.name));
        for c in &mut self.children {
            c.sort();
        }
    }
}

fn build_tree(folded: &str) -> Node {
    let mut root = Node {
        name: "all".to_string(),
        total: 0,
        children: Vec::new(),
    };
    for line in folded.lines() {
        let Some((stack, count)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(count) = count.trim().parse::<u64>() else {
            continue;
        };
        root.total += count;
        let mut node = &mut root;
        for frame in stack.split(';') {
            node = node.child(frame);
            node.total += count;
        }
    }
    root.sort();
    root
}

/// Layout and appearance. Fixed rather than configurable: these are the values
/// this project's flamegraphs have always used, and a chart whose geometry
/// varies between runs cannot be compared against an older one by eye.
const WIDTH: f64 = 1200.0;
const FRAME_HEIGHT: f64 = 16.0;
const FONT_SIZE: f64 = 12.0;
const PAD_TOP: f64 = 54.0;
const PAD_BOTTOM: f64 = 20.0;
/// Frames narrower than this are dropped — below it a frame is a sliver that
/// cannot be read or clicked, and keeping them is what turns a flamegraph into
/// a multi-megabyte file.
const MIN_WIDTH_PX: f64 = 0.8;

/// Render folded stacks to a self-contained SVG.
pub fn render(folded: &str, title: &str, subtitle: &str) -> String {
    let title = fit_title(title);
    let root = build_tree(folded);
    let depth = tree_depth(&root);
    let height = PAD_TOP + PAD_BOTTOM + (depth as f64) * FRAME_HEIGHT;

    let mut body = String::new();
    if root.total > 0 {
        emit(&root, 0.0, 0, root.total, height, &mut body);
    }

    format!(
        r##"<?xml version="1.0" standalone="no"?>
<svg version="1.1" width="{WIDTH}" height="{height:.0}" viewBox="0 0 {WIDTH} {height:.0}" xmlns="http://www.w3.org/2000/svg" onload="init(evt)">
<style type="text/css">
  text {{ font-family: Verdana, Helvetica, sans-serif; font-size: {FONT_SIZE}px; fill: #000; }}
  #title {{ font-size: 17px; text-anchor: middle; }}
  #subtitle, #details, #hint {{ font-size: 11px; fill: #555; }}
  .frame rect {{ stroke: #fff; stroke-width: 0.4; }}
  .frame:hover rect {{ stroke: #000; stroke-width: 0.7; }}
  .hidden {{ display: none; }}
</style>
<rect x="0" y="0" width="{WIDTH}" height="{height:.0}" fill="#f8f8f2"/>
<text id="title" x="{half:.0}" y="24">{title}</text>
<text id="subtitle" x="8" y="40">{subtitle}</text>
<text id="hint" x="{WIDTH}" y="40" text-anchor="end">click a frame to zoom · click the title to reset · Ctrl-F to search</text>
<text id="details" x="8" y="{details_y:.0}"> </text>
<g id="frames">
{body}</g>
<script type="text/ecmascript"><![CDATA[
{script}
]]></script>
</svg>
"##,
        half = WIDTH / 2.0,
        details_y = height - 6.0,
        title = escape(&title),
        subtitle = escape(subtitle),
        script = SCRIPT,
    )
}

/// A title too wide for the chart, shortened from the middle.
///
/// A default title carries the model id, which for a locally-loaded GGUF is an
/// absolute path — long enough to run off both edges of a centred heading and
/// take the sample count with it. The *tail* is the informative part (the file
/// name), so the middle is what gives way.
fn fit_title(title: &str) -> String {
    const MAX: usize = 96;
    let chars: Vec<char> = title.chars().collect();
    if chars.len() <= MAX {
        return title.to_string();
    }
    let head: String = chars[..MAX / 3].iter().collect();
    let tail: String = chars[chars.len() - (MAX - MAX / 3 - 1)..].iter().collect();
    format!("{head}…{tail}")
}

fn tree_depth(node: &Node) -> usize {
    1 + node.children.iter().map(tree_depth).max().unwrap_or(0)
}

/// Emit one frame and its children. `x` is in samples-as-pixels of the full
/// width; depth grows downward from the top, so the root is the bottom row —
/// the "icicle" orientation this project's existing flamegraphs use.
fn emit(node: &Node, x: f64, depth: usize, total: u64, height: f64, out: &mut String) {
    let width = node.total as f64 / total as f64 * WIDTH;
    if width < MIN_WIDTH_PX {
        return;
    }
    let y = height - PAD_BOTTOM - ((depth + 1) as f64) * FRAME_HEIGHT;
    let (r, g, b) = color(&node.name);
    let pct = node.total as f64 * 100.0 / total as f64;

    let _ = write!(
        out,
        r#"<g class="frame"><title>{name} ({count} samples, {pct:.2}%)</title>"#,
        name = escape(&node.name),
        count = node.total,
    );
    let _ = write!(
        out,
        r#"<rect x="{x:.1}" y="{y:.1}" width="{width:.1}" height="{h}" fill="rgb({r},{g},{b})"/>"#,
        h = FRAME_HEIGHT - 1.0,
    );
    // Roughly 0.59 em per character at this font; a label that would overflow
    // its frame is truncated with an ellipsis rather than clipped, so a narrow
    // frame shows *something* readable.
    let room = ((width - 6.0) / (FONT_SIZE * 0.59)) as usize;
    if room >= 3 {
        let label = if node.name.chars().count() > room {
            format!("{}…", node.name.chars().take(room - 1).collect::<String>())
        } else {
            node.name.clone()
        };
        let _ = write!(
            out,
            r#"<text x="{tx:.1}" y="{ty:.1}">{label}</text>"#,
            tx = x + 3.0,
            ty = y + FRAME_HEIGHT - 4.5,
            label = escape(&label),
        );
    }
    out.push_str("</g>\n");

    let mut cx = x;
    for child in &node.children {
        emit(child, cx, depth + 1, total, height, out);
        cx += child.total as f64 / total as f64 * WIDTH;
    }
}

/// A frame's fill colour: the warm palette, varied deterministically by name so
/// the same function is the same colour in two profiles and neighbouring frames
/// are distinguishable. Hue carries no meaning, which is the point — a
/// flamegraph encodes everything in width.
fn color(name: &str) -> (u8, u8, u8) {
    // A cheap string hash spread over the low bits; `hash` only has to be
    // stable and well-mixed, not cryptographic.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    let v1 = (hash & 0xffff) as f64 / 65535.0;
    let v2 = ((hash >> 16) & 0xffff) as f64 / 65535.0;
    let v3 = ((hash >> 32) & 0xffff) as f64 / 65535.0;
    (
        (205.0 + 50.0 * v3) as u8,
        (30.0 + 200.0 * v1) as u8,
        (30.0 + 45.0 * v2) as u8,
    )
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

/// Zoom, reset and search. Kept small and dependency-free: it reads the frames
/// already in the document rather than carrying a second copy of the tree.
const SCRIPT: &str = r#"
var frames = [], details = null;
function init() {
  frames = Array.prototype.slice.call(document.querySelectorAll('#frames g.frame'));
  details = document.getElementById('details');
  frames.forEach(function (g) {
    var rect = g.querySelector('rect');
    // Stash the unzoomed geometry so reset is exact rather than a reload.
    rect.setAttribute('data-x0', rect.getAttribute('x'));
    rect.setAttribute('data-w0', rect.getAttribute('width'));
    rect.setAttribute('data-fill', rect.getAttribute('fill'));
    g.addEventListener('click', function (e) { e.stopPropagation(); zoom(g); });
    g.addEventListener('mouseover', function () { details.textContent = label(g); });
    g.addEventListener('mouseout', function () { details.textContent = ' '; });
  });
  document.getElementById('title').addEventListener('click', reset);
  window.addEventListener('keydown', function (e) {
    if (e.key === 'f' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); search(); }
    if (e.key === 'Escape') { clearSearch(); reset(); }
  });
}
function label(g) { return g.querySelector('title').textContent; }
function name(g) { return label(g).replace(/ \(\d+ samples.*$/, ''); }
function geom(g) {
  var r = g.querySelector('rect');
  return { x: parseFloat(r.getAttribute('data-x0')), w: parseFloat(r.getAttribute('data-w0')),
           y: parseFloat(r.getAttribute('y')) };
}
function place(g, x, w) {
  var rect = g.querySelector('rect'), text = g.querySelector('text');
  rect.setAttribute('x', x); rect.setAttribute('width', w);
  if (!text) return;
  text.setAttribute('x', x + 3);
  var full = name(g), room = Math.floor((w - 6) / (12 * 0.59));
  text.textContent = room < 3 ? '' : (full.length > room ? full.slice(0, room - 1) + '\u2026' : full);
}
function zoom(g) {
  var b = geom(g);
  if (!(b.w > 0)) return;
  var scale = 1200 / b.w;
  frames.forEach(function (f) {
    var fb = geom(f);
    // Inside the zoomed subtree means: within its horizontal span and at or
    // below its row. A frame outside it is hidden, not moved off-canvas — an
    // off-canvas frame still answers hit tests and tooltips.
    var inside = fb.x >= b.x - 0.01 && fb.x + fb.w <= b.x + b.w + 0.01 && fb.y <= b.y;
    f.classList.toggle('hidden', !inside);
    if (inside) place(f, (fb.x - b.x) * scale, fb.w * scale);
  });
  details.textContent = 'zoomed: ' + name(g);
}
function reset() {
  frames.forEach(function (f) {
    f.classList.remove('hidden');
    var fb = geom(f);
    place(f, fb.x, fb.w);
  });
  details.textContent = ' ';
}
function search() {
  var term = prompt('Highlight frames matching:');
  if (term === null) return;
  if (term === '') { clearSearch(); return; }
  // Matched samples are summed over the *shallowest* matching frames only, so
  // a function and its matching callees are not counted twice.
  var matched = 0, total = 0, hits = [];
  frames.forEach(function (f) {
    var rect = f.querySelector('rect'), n = samples(f), g = geom(f);
    if (g.x === 0 && f === frames[0]) total = n;
    if (label(f).indexOf(term) !== -1) {
      rect.setAttribute('fill', 'rgb(30,120,220)');
      hits.push(g);
    } else {
      rect.setAttribute('fill', rect.getAttribute('data-fill'));
    }
  });
  hits.forEach(function (g, i) {
    for (var j = 0; j < hits.length; j++) {
      var o = hits[j];
      if (j !== i && o.y > g.y && o.x <= g.x && o.x + o.w >= g.x + g.w) return;
    }
    matched += g.w;
  });
  details.textContent = 'matched "' + term + '": ' + (100 * matched / 1200).toFixed(2) + '% of samples';
}
function samples(g) {
  var m = label(g).match(/\((\d+) samples/);
  return m ? parseInt(m[1], 10) : 0;
}
function clearSearch() {
  frames.forEach(function (f) {
    var rect = f.querySelector('rect');
    rect.setAttribute('fill', rect.getAttribute('data-fill'));
  });
  details.textContent = ' ';
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
orangu-server 1234 [003] 98.765432:     250000 cycles:ppp:
\t    7f0a1234abcd dot_avx2+0x1c (/opt/orangu-server)
\t    7f0a1234ab00 run_layers+0x8 (/opt/orangu-server)

orangu-server 1234 [003] 98.765500:     250000 cycles:ppp:
\t    7f0a1234abcd dot_avx2+0x1c (/opt/orangu-server)
\t    7f0a1234ab00 run_layers+0x8 (/opt/orangu-server)

orangu-server 1234 [003] 98.765600:     250000 cycles:ppp:
\tffffffff81234567 read_hpet+0x59 ([kernel.kallsyms])
\t    7f0a1234ab00 run_layers+0x8 (/opt/orangu-server)
";

    #[test]
    fn collapse_reverses_the_stack_and_counts_identical_ones_together() {
        let out = collapse(SAMPLE);
        assert_eq!(out["orangu-server;run_layers;dot_avx2"], 2);
        assert_eq!(out["orangu-server;run_layers;read_hpet_[k]"], 1);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_kernel_frame_is_marked_from_its_dso_not_its_name() {
        // `read_hpet` is an ordinary-looking symbol that lives in the kernel.
        // Guessing from the name files it as application code, which is what
        // the dso field is read to avoid.
        let out = collapse(SAMPLE);
        assert!(out.keys().any(|k| k.ends_with("read_hpet_[k]")));
        assert!(!out.keys().any(|k| k.contains("dot_avx2_[k]")));
    }

    #[test]
    fn a_symbol_containing_parentheses_keeps_them_and_still_finds_its_dso() {
        let line = "\t    7f0a1234abcd std::function<void ()>::operator()+0x1c (/opt/x.so)";
        assert_eq!(
            parse_frame(line).as_deref(),
            Some("std::function<void ()>::operator()")
        );
    }

    #[test]
    fn a_frame_with_no_dso_is_still_a_frame() {
        assert_eq!(
            parse_frame("\t 7f0a1234abcd [unknown]").as_deref(),
            Some("[unknown]")
        );
    }

    #[test]
    fn a_plus_inside_a_symbol_is_not_mistaken_for_an_offset() {
        let line = "\t 7f00 core::ops::Add<a+b>::add (/opt/x)";
        assert_eq!(
            parse_frame(line).as_deref(),
            Some("core::ops::Add<a+b>::add")
        );
    }

    #[test]
    fn a_comm_containing_a_space_is_not_truncated_at_it() {
        // The comm is not the first field; taking it as such loses half the
        // thread name and splits one thread's samples across two roots.
        let script = "my worker thread 991/992 [000] 1.0: 1 cycles:\n\t 7f00 f (/x)\n";
        let out = collapse(script);
        assert_eq!(out.keys().next().unwrap(), "my worker thread;f");
    }

    #[test]
    fn cycles_are_folded_and_every_sample_survives() {
        let script = "\
w 1 [0] 1.0: 1 cycles:
\t 1 work (/x)
\t 2 join (/x)
\t 3 helper (/x)
\t 4 join (/x)
\t 5 helper (/x)
\t 6 root (/x)

w 1 [0] 2.0: 1 cycles:
\t 1 work (/x)
\t 2 join (/x)
\t 3 helper (/x)
\t 4 root (/x)
";
        let out = collapse(script);
        assert_eq!(out.len(), 1, "both split depths must fold to one stack");
        assert_eq!(out["w;root;helper;join;work"], 2);
    }

    #[test]
    fn a_semicolon_in_a_symbol_cannot_forge_a_stack_separator() {
        let line = "\t 7f00 weird;name (/x)";
        assert_eq!(parse_frame(line).as_deref(), Some("weird:name"));
    }

    #[test]
    fn the_rendered_svg_is_self_contained_and_deterministic() {
        let folded = "a;b 3\na;c 1\n";
        let one = render(folded, "t", "s");
        assert_eq!(one, render(folded, "t", "s"));
        assert!(one.starts_with("<?xml"));
        // No external references at all: a strict viewer must need nothing else.
        assert!(!one.contains("http://www.w3.org/1999/xlink"));
        assert!(!one.contains("<image"));
        assert!(one.contains("<title>b (3 samples, 75.00%)</title>"));
    }

    #[test]
    fn a_frame_below_the_minimum_width_is_dropped_rather_than_drawn_as_a_sliver() {
        // One sample in ten thousand is 0.12 px at 1200 px wide.
        let mut folded = String::from("a;wide 9999\n");
        folded.push_str("a;narrow 1\n");
        let svg = render(&folded, "t", "s");
        assert!(svg.contains(">wide"));
        assert!(!svg.contains("narrow"));
    }

    #[test]
    fn an_over_wide_title_is_shortened_from_the_middle_keeping_the_file_name() {
        let long = "/mnt/ai/jews/models/models--bartowski--Llama-3.2-1B-Instruct-GGUF/\
                    snapshots/067b946cf014b7c697f3654f621d577a3e3afd1c/\
                    Llama-3.2-1B-Instruct-Q4_K_M.gguf · prefill pp 1024";
        let fitted = fit_title(long);
        assert!(fitted.chars().count() <= 96);
        assert!(fitted.ends_with("prefill pp 1024"), "{fitted}");
        assert!(fitted.starts_with("/mnt/ai"), "{fitted}");
        assert!(fitted.contains('…'));
        assert_eq!(fit_title("short"), "short");
    }

    #[test]
    fn markup_in_a_symbol_name_is_escaped() {
        let svg = render("a;Vec<T>&x 1\n", "t", "s");
        assert!(svg.contains("Vec&lt;T&gt;&amp;x"));
        assert!(!svg.contains("Vec<T>&x"));
    }
}

#[cfg(test)]
mod xcheck {
    /// Cross-check against the Perl pipeline on a real capture, when one has
    /// been left at `ORANGU_XCHECK_SCRIPT` / `ORANGU_XCHECK_FOLDED`.
    /// `#[ignore]`d: it needs a `perf script` dump and a FlameGraph checkout,
    /// neither of which a normal test run has.
    #[test]
    #[ignore]
    fn native_collapse_agrees_with_the_perl_pipeline() {
        let Ok(script) = std::env::var("ORANGU_XCHECK_SCRIPT") else {
            eprintln!("skipping: set ORANGU_XCHECK_SCRIPT and ORANGU_XCHECK_FOLDED");
            return;
        };
        let perl = std::env::var("ORANGU_XCHECK_FOLDED").expect("ORANGU_XCHECK_FOLDED");
        let ours = super::collapse(&std::fs::read_to_string(script).unwrap());
        let theirs = std::fs::read_to_string(perl).unwrap();

        let their_total: u64 = theirs
            .lines()
            .filter_map(|l| l.rsplit_once(' '))
            .filter_map(|(_, c)| c.trim().parse::<u64>().ok())
            .sum();
        let our_total: u64 = ours.values().sum();
        // Sample conservation is the strongest available check: the two
        // implementations may name a frame differently, but neither may invent
        // or lose a sample.
        assert_eq!(our_total, their_total, "sample totals must match exactly");
        eprintln!(
            "native: {our_total} samples in {} stacks; perl: {their_total} in {} stacks",
            ours.len(),
            theirs.lines().count()
        );
    }
}
