// The orangu-bench console. Vanilla JS, no build step, no framework — the
// same shape as orangu-server's own web UI, and for the same reason: this
// file is served straight out of the binary.
//
// It owns exactly two things: turning the form into a run definition, and
// drawing what came back. Nothing here decides what a benchmark *is* — the
// server builds the command line, and the numbers are read out of the bundle
// the run wrote.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ---- theme ---------------------------------------------------------

  const themeToggleBtn = $("theme-toggle-btn");
  const THEME_KEY = "orangu-theme";

  function effectiveTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved) return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function paintThemeToggle() {
    const dark = effectiveTheme() === "dark";
    const label = dark ? "Switch to light mode" : "Switch to dark mode";
    themeToggleBtn.textContent = dark ? "☀️" : "🌙";
    themeToggleBtn.setAttribute("aria-label", label);
    themeToggleBtn.setAttribute("title", label);
  }

  themeToggleBtn.addEventListener("click", () => {
    localStorage.setItem(THEME_KEY, effectiveTheme() === "dark" ? "light" : "dark");
    document.documentElement.setAttribute("data-theme", effectiveTheme());
    paintThemeToggle();
  });

  // ---- the modes -----------------------------------------------------

  // What each measurement calls its inputs, and what it measures. The help
  // line is not decoration: "prefill" and "decode" are different numbers in
  // different units, and picking the wrong one is the most expensive mistake
  // this form allows.
  const MODES = {
    "tg": {
      points: "Context depths",
      gen: "Tokens to generate",
      help: "Steady-state decode, timed from the first streamed token to the last — prefill and time-to-first-token excluded.",
    },
    "pp": {
      points: "Prompt lengths",
      help: "Prompt-processing throughput, from the server's own timings — a prefix-cache hit is reported, not disguised as a fast run.",
    },
    "pg": {
      points: "Prompt lengths",
      gen: "Tokens to generate",
      help: "Prefill and generation in one request, reported as (prompt + generated) over the total time — the whole turn, which is what a user waits for.",
    },
    "pp-continue": {
      points: "Added tokens",
      base: true,
      help: "Prefill of a continuation on top of a cached base: the narrow-batch regime, where per-forward overhead dominates.",
    },
    "curve": {
      gen: "Tokens in the pass",
      bucket: true,
      help: "One generation, decode rate bucketed by context position — decode-vs-context scaling without a deep-context prefill.",
    },
    "streams": {
      points: "Concurrent streams",
      gen: "Tokens per stream",
      help: "Aggregate tok/s across N simultaneous streams: whether the engine can fill the device.",
    },
    "embed": {
      points: "Prompt lengths",
      help: "Forward-pass throughput against /v1/embeddings — the only mode an embedding-only server answers at all.",
    },
    "decode-cpu": {
      points: "Context depths",
      gen: "Tokens to generate",
      help: "The server's own CPU milliseconds per generated token, with prefill excluded. Reported in ms/token, not tok/s.",
    },
  };

  // ---- form ----------------------------------------------------------

  const SPEC_KEY = "orangu-bench-spec";
  const PRESET_KEY = "orangu-bench-preset";
  const RUN_KEY = "orangu-bench-run";
  const state = {
    id: null,
    logCount: 0,
    timer: null,
    frameSrc: null,
    capabilities: {},
    // The scaling sweeps the server offers, one per measurement.
    presets: [],
    // Every run this console knows about, for the comparison menu, and the
    // last view drawn, so that menu can be refreshed without re-fetching it.
    runs: [],
    view: null,
  };

  function specFromForm() {
    const pid = $("f-pid").value.trim();
    return {
      url: $("f-url").value.trim(),
      model: $("f-model").value.trim(),
      label: $("f-label").value.trim(),
      mode: $("f-mode").value,
      points: $("f-points").value.trim(),
      n_gen: num($("f-gen").value, 128),
      reps: num($("f-reps").value, 3),
      bucket: num($("f-bucket").value, 256),
      pp_continue_base: num($("f-base").value, 512),
      timeout: num($("f-timeout").value, 600),
      // 0 is a legitimate value here — "no delay" — so it cannot go through
      // `num`, which treats 0 as "not a number, use the default".
      delay: Math.max(0, Number.parseInt($("f-delay").value, 10) || 0),
      warmup: $("f-warmup").checked,
      flamegraph: $("f-flamegraph").checked,
      flamegraph_freq: num($("f-freq").value, 999),
      flamegraph_call_graph: $("f-call-graph").value,
      flamegraph_pid: pid === "" ? null : num(pid, 0),
      chart: $("f-chart").checked,
    };
  }

  function applySpec(spec) {
    if (!spec) return;
    $("f-url").value = spec.url ?? "";
    $("f-model").value = spec.model ?? "";
    $("f-label").value = spec.label ?? "";
    $("f-mode").value = MODES[spec.mode] ? spec.mode : "tg";
    $("f-points").value = spec.points ?? "0";
    $("f-gen").value = spec.n_gen ?? 128;
    $("f-reps").value = spec.reps ?? 3;
    $("f-bucket").value = spec.bucket ?? 256;
    $("f-base").value = spec.pp_continue_base ?? 512;
    $("f-timeout").value = spec.timeout ?? 600;
    $("f-delay").value = spec.delay ?? 0;
    $("f-warmup").checked = spec.warmup !== false;
    $("f-chart").checked = spec.chart !== false;
    $("f-flamegraph").checked = !!spec.flamegraph;
    $("f-freq").value = spec.flamegraph_freq ?? 999;
    $("f-call-graph").value = spec.flamegraph_call_graph ?? "fp";
    $("f-pid").value = spec.flamegraph_pid ?? "";
    paintMode();
  }

  function num(value, fallback) {
    const n = Number.parseInt(value, 10);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  }

  // Show the fields this measurement actually has, and name them for it —
  // "Context depths" and "Concurrent streams" are the same input only in the
  // sense that both are a list of numbers.
  function paintMode() {
    const mode = MODES[$("f-mode").value] ?? MODES.tg;
    $("mode-help").textContent = mode.help;
    $("wrap-points").hidden = !mode.points;
    $("points-label").textContent = mode.points ?? "";
    $("wrap-gen").hidden = !mode.gen;
    $("gen-label").textContent = mode.gen ?? "";
    $("wrap-bucket").hidden = !mode.bucket;
    $("wrap-base").hidden = !mode.base;
    paintPresets();
  }

  // The scaling tests available for the selected measurement, named by the
  // range they sweep — "0 to 4096" under Prefill. Rebuilt whenever the
  // measurement changes, because a decode sweep's depths are not a
  // concurrency sweep's stream counts.
  function paintPresets(select) {
    const menu = $("f-preset");
    const previous = select ?? "";
    menu.replaceChildren();
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "None — choose the points yourself";
    menu.appendChild(none);
    for (const preset of state.presets.filter((p) => p.mode === $("f-mode").value)) {
      const option = document.createElement("option");
      option.value = preset.range;
      option.textContent = preset.range;
      menu.appendChild(option);
    }
    // A preset only survives a measurement change if the new measurement has
    // one by that name; otherwise the sweep on screen would not be the sweep
    // selected.
    menu.value = [...menu.options].some((o) => o.value === previous) ? previous : "";
    applyPreset();
  }

  // Fill in and lock what the chosen preset owns, so the sweep about to run is
  // on screen rather than implied. "None" hands every field back.
  function applyPreset() {
    const chosen = state.presets.find(
      (p) => p.mode === $("f-mode").value && p.range === $("f-preset").value,
    );
    localStorage.setItem(PRESET_KEY, chosen ? chosen.range : "");
    $("preset-help").textContent = chosen ? chosen.about : "";
    $("preset-help").hidden = !chosen;
    if (chosen) {
      $("f-points").value = chosen.points;
      $("f-gen").value = chosen.n_gen;
      $("f-reps").value = chosen.reps;
      $("f-bucket").value = chosen.bucket;
      $("f-base").value = chosen.pp_continue_base;
    }
    for (const id of ["f-points", "f-gen", "f-reps", "f-bucket", "f-base"]) {
      $(id).disabled = !!chosen;
    }
  }

  function paintFlamegraph() {
    const on = $("f-flamegraph").checked;
    $("flamegraph-options").hidden = !on;
    // The two ways a profile can disappoint after the run rather than before
    // it, said next to the checkbox that promises it.
    const notes = [];
    if (on && state.capabilities.have_perf === false) {
      notes.push("`perf` was not found on PATH — the run will measure, but no profile will be recorded.");
    }
    if (on && state.capabilities.have_rsvg === false) {
      notes.push("`rsvg-convert` was not found on PATH — the SVG will be written, the PNG will not.");
    }
    if (on && !$("f-warmup").checked) {
      notes.push("A profile needs the warmup: `perf record -p` never picks up threads created after it attaches, and a server builds its compute threads on the first request.");
    }
    $("flamegraph-note").textContent = notes.join(" ");
    $("flamegraph-note").hidden = notes.length === 0;
  }

  $("f-mode").addEventListener("change", paintMode);
  $("f-preset").addEventListener("change", applyPreset);
  $("f-flamegraph").addEventListener("change", paintFlamegraph);
  $("f-warmup").addEventListener("change", paintFlamegraph);

  // ---- running -------------------------------------------------------

  $("run-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const spec = specFromForm();
    localStorage.setItem(SPEC_KEY, JSON.stringify(spec));
    showError(null);
    $("run-btn").disabled = true;
    try {
      const res = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(spec),
      });
      const body = await res.json();
      if (!res.ok) {
        showError(body.error ?? `the run was refused (HTTP ${res.status})`);
        return;
      }
      attach(body.id);
    } catch (e) {
      showError(`could not reach the console: ${e}`);
    } finally {
      $("run-btn").disabled = false;
    }
  });

  $("cancel-btn").addEventListener("click", async () => {
    if (!state.id) return;
    $("cancel-btn").disabled = true;
    try {
      await fetch(`/api/runs/${encodeURIComponent(state.id)}/cancel`, { method: "POST" });
    } catch {
      // The poll below reports the run's real state either way.
    }
    $("cancel-btn").disabled = false;
  });

  function showError(message) {
    $("form-error").textContent = message ?? "";
    $("form-error").hidden = !message;
  }

  // Every result section, so switching runs and deleting one hide the same
  // set — a stale flamegraph left on screen under a new run's summary would
  // be a wrong answer, not a cosmetic slip.
  const SECTIONS = [
    "summary-section",
    "compare-section",
    "flamegraph-section",
    "chart-section",
  ];

  // Switch the results pane to a run and follow it. Log lines are fetched
  // from `logCount` on, so a long run's log is sent once rather than on
  // every poll.
  function attach(id) {
    if (state.timer) clearInterval(state.timer);
    state.id = id;
    state.logCount = 0;
    state.frameSrc = null;
    state.view = null;
    // Remembered so a reload comes back to the run you were reading rather
    // than to whichever happens to be newest — the two differ as soon as you
    // open an older run to compare against.
    localStorage.setItem(RUN_KEY, id);
    $("log").textContent = "";
    $("empty").hidden = true;
    for (const s of SECTIONS) $(s).hidden = true;
    poll();
    state.timer = setInterval(poll, 700);
  }

  // **New**: put the results pane back to its empty state and let go of the
  // run it was showing. The form is deliberately left as it is — a new run is
  // almost always the last one with one field changed, which is exactly what
  // an A/B arm is.
  function newRun() {
    stopPolling();
    state.id = null;
    state.view = null;
    state.frameSrc = null;
    state.logCount = 0;
    // Empty, not removed: "the user cleared this" and "the user has never
    // opened a run" are different, and only the second should reopen the
    // newest run on the next load.
    localStorage.setItem(RUN_KEY, "");
    $("log").textContent = "";
    $("run-status").hidden = true;
    $("log-section").hidden = true;
    for (const s of SECTIONS) $(s).hidden = true;
    $("empty").hidden = false;
    $("run-btn").disabled = false;
    $("cancel-btn").hidden = true;
    // Back to empty, not to a word: the topbar's job here is to name the
    // server being measured, and there isn't one yet.
    $("target").textContent = "";
    showError(null);
    loadHistory();
  }

  $("new-run-btn").addEventListener("click", newRun);

  async function poll() {
    if (!state.id) return;
    try {
      const res = await fetch(
        `/api/runs/${encodeURIComponent(state.id)}?from=${state.logCount}`,
        { cache: "no-store" },
      );
      if (!res.ok) {
        stopPolling();
        return;
      }
      render(await res.json());
    } catch {
      // A console being restarted underneath the page: keep polling, the
      // next one succeeds.
    }
  }

  function stopPolling() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }

  // ---- drawing the result --------------------------------------------

  function render(view) {
    state.view = view;
    const running = view.status === "running";
    $("empty").hidden = true;
    $("run-status").hidden = false;
    $("cancel-btn").hidden = !running;
    $("run-btn").disabled = running;

    $("status-badge").className = `badge ${view.status}`;
    $("status-badge").textContent = view.status;
    $("status-text").textContent = describe(view.spec);
    $("target").textContent = view.spec?.url ?? "";

    const meta = [];
    if (view.started) meta.push(new Date(view.started * 1000).toLocaleString());
    if (view.seconds > 0) meta.push(`${view.seconds.toFixed(1)}s`);
    if (running) meta.push("measuring…");
    $("status-meta").textContent = meta.join(" · ");

    appendLog(view.log ?? []);
    state.logCount = view.log_total ?? state.logCount;
    $("log-section").hidden = false;
    saveTarget($("log-dl"), view, "log.txt", (view.artifacts ?? []).includes("log.txt"));

    renderSummary(view);
    renderCompare(view);
    renderFlamegraph(view);
    renderChart(view);

    if (!running) {
      stopPolling();
      loadHistory();
    }
  }

  function describe(spec) {
    if (!spec) return "";
    const points = MODES[spec.mode]?.points ? ` · ${spec.points}` : "";
    return `${modeName(spec.mode)}${points}`;
  }

  // The measurement's name as the form says it. Taken from the <option> rather
  // than kept in a second list here, so the run being read back and the run
  // being defined are never described in two different vocabularies.
  function modeName(mode) {
    const option = $("f-mode").querySelector(`option[value="${mode}"]`);
    return option ? option.textContent : (mode ?? "?");
  }

  function appendLog(lines) {
    if (!lines.length) return;
    const log = $("log");
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    for (const line of lines) {
      const span = document.createElement("span");
      if (line.err) span.className = "err";
      span.textContent = `${line.text}\n`;
      log.appendChild(span);
    }
    // Follow the tail only if the reader was already at it — scrolling back
    // to read a row while a sweep continues must not be undone every 700ms.
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  function renderSummary(view) {
    const records = view.records ?? [];
    if (!records.length) {
      $("summary-section").hidden = true;
      return;
    }
    $("summary-section").hidden = false;
    const artifacts = view.artifacts ?? [];
    // The report is built on demand, so the button is offered as soon as the
    // run has the bundle it would be built from.
    $("report-btn").hidden = !artifacts.includes("bundle.json");
    saveTarget($("bundle-dl"), view, "bundle.json", artifacts.includes("bundle.json"));

    const env = $("summary-env");
    env.replaceChildren();
    const props = view.props ?? {};
    const host = view.host ?? {};
    const pairs = [
      ["label", records[0].label],
      ["model", props.model],
      ["backend", props.backend],
      ["server", props.pid ? `pid ${props.pid}, up ${props.uptime_seconds ?? "?"}s` : null],
      ["host", [host.os, host.arch].filter(Boolean).join("/") || null],
      ["clocks", (host.clocks ?? []).map((c) => `${c.card} ${c.sclk} (${c.power_level})`).join(", ") || null],
      ["url", view.run?.url],
    ];
    for (const [key, value] of pairs) {
      if (!value) continue;
      const k = document.createElement("span");
      k.className = "k";
      k.textContent = key;
      const v = document.createElement("span");
      v.className = "v";
      v.textContent = value;
      env.append(k, v);
    }

    const table = $("summary-table");
    table.replaceChildren();
    const head = table.createTHead().insertRow();
    for (const [text, cls] of [
      ["measurement", "left"], ["n", ""], ["best", ""], ["mean", ""], ["± sd (n−1)", ""], ["unit", "left"],
    ]) {
      const th = document.createElement("th");
      th.textContent = text;
      if (cls) th.className = cls;
      head.appendChild(th);
    }
    const body = table.createTBody();
    for (const r of records) {
      const row = body.insertRow();
      // `cpu` is milliseconds per token, not tokens per second — the one
      // mode whose column is not a rate, and it must not be read as one.
      const unit = r.mode === "cpu" ? "ms/token" : "tok/s";
      cell(row, MEASUREMENT[r.mode] ?? r.mode, "left");
      cell(row, String(r.n), "num");
      cell(row, r.best.toFixed(2), "num best");
      cell(row, r.mean.toFixed(2), "num");
      // The sample estimator (÷ n−1), which is what other benchmarks report
      // and therefore the only one worth putting a number beside. A single
      // repetition has none: "—", never "0.00", which would claim a spread
      // was measured and found to be zero. Older bundles have no such field.
      cell(row, r.sd_sample == null ? "—" : r.sd_sample.toFixed(2), "num");
      cell(row, unit, "left");
    }
  }

  // What a record's `mode` means in words. The bundle stores the short form
  // (it is a data file); the table has room to say it.
  const MEASUREMENT = {
    tg: "decode",
    pp: "prefill",
    curve: "decode @ context",
    cpu: "decode CPU",
    embed: "embedding",
  };

  function cell(row, text, className) {
    const td = row.insertCell();
    td.textContent = text;
    if (className) td.className = className;
    return td;
  }

  // "Is this build faster than the one from Tuesday?" — the question a
  // benchmark is usually run to answer, and the reason every run archives a
  // bundle. Offered as soon as this run has one of its own.
  // **Report**: ask the console to build this run's PDF, then save it. The
  // build takes a moment (it lays out a document and embeds two images), so
  // the button says what it is doing rather than appearing to have missed the
  // click.
  $("report-btn").addEventListener("click", async () => {
    if (!state.id) return;
    const button = $("report-btn");
    const label = button.querySelector("span");
    const was = label.textContent;
    button.disabled = true;
    label.textContent = "Building…";
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(state.id)}/report`, {
        method: "POST",
      });
      const body = await res.json();
      if (!res.ok) {
        showError(body.error ?? `the report was not built (HTTP ${res.status})`);
        return;
      }
      // A plain anchor click, so what lands on disk is the file the console
      // wrote — same path every other save here takes.
      const link = $("report-dl");
      saveTarget(link, { id: state.id }, "report.pdf", true);
      link.click();
    } catch (e) {
      showError(`could not reach the console: ${e}`);
    } finally {
      label.textContent = was;
      button.disabled = false;
    }
  });

  function renderCompare(view) {
    const ready = (view.artifacts ?? []).includes("bundle.json");
    $("compare-section").hidden = !ready;
    if (!ready) return;

    const menu = $("compare-with");
    const chosen = view.compare?.with ?? "";
    const options = state.runs.filter((r) => r.id !== view.id && r.has_bundle);
    // Rebuilt only when the set of comparable runs changes, so an open menu
    // is not yanked out from under a click by the poll behind it.
    const signature = `${view.id}|${options.map((r) => r.id).join(",")}`;
    if (menu.dataset.signature !== signature) {
      menu.dataset.signature = signature;
      menu.replaceChildren();
      const none = document.createElement("option");
      none.value = "";
      none.textContent = options.length
        ? "Compare with an earlier run…"
        : "No earlier run to compare with";
      menu.appendChild(none);
      for (const run of options) {
        const option = document.createElement("option");
        option.value = run.id;
        const when = run.started ? new Date(run.started * 1000).toLocaleString() : run.id;
        const label = run.spec?.label ? `${run.spec.label} · ` : "";
        option.textContent = `${label}${modeName(run.spec?.mode)} · ${when}`;
        menu.appendChild(option);
      }
      menu.disabled = options.length === 0;
    }
    if (menu.value !== chosen) menu.value = chosen;

    const done = !!view.compare;
    $("compare-result").hidden = !done;
    if (!done) return;
    $("compare-text").textContent = view.compare.text ?? "";
    const artifacts = view.artifacts ?? [];
    const hasChart = artifacts.includes("compare.svg");
    $("compare-img").hidden = !hasChart;
    if (hasChart) $("compare-img").src = artifactUrl(view.id, "compare.svg");
    saveTarget($("compare-svg-dl"), view, "compare.svg", hasChart);
    saveTarget($("compare-png-dl"), view, "compare.png", artifacts.includes("compare.png"));
    saveTarget($("compare-pdf-dl"), view, "compare.pdf", artifacts.includes("compare.pdf"));
    saveTarget($("compare-txt-dl"), view, "compare.txt", artifacts.includes("compare.txt"));
  }

  $("compare-with").addEventListener("change", async () => {
    const other = $("compare-with").value;
    if (!state.id || !other) return;
    $("compare-error").hidden = true;
    $("compare-with").disabled = true;
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(state.id)}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ with: other }),
      });
      const body = await res.json();
      if (!res.ok) {
        $("compare-error").textContent = body.error ?? `HTTP ${res.status}`;
        $("compare-error").hidden = false;
      }
    } catch (e) {
      $("compare-error").textContent = `could not reach the console: ${e}`;
      $("compare-error").hidden = false;
    } finally {
      $("compare-with").disabled = false;
    }
    // The comparison is written to disk, so the run view is where it is read
    // back from — one path for a fresh comparison and for a reloaded one.
    poll();
  });

  function renderFlamegraph(view) {
    const artifacts = view.artifacts ?? [];
    const has = artifacts.includes("flamegraph.svg");
    $("flamegraph-section").hidden = !has;
    if (!has) return;

    const svg = artifactUrl(view.id, "flamegraph.svg");
    saveTarget($("flamegraph-svg-dl"), view, "flamegraph.svg", true);
    saveTarget(
      $("flamegraph-png-dl"),
      view,
      "flamegraph.png",
      artifacts.includes("flamegraph.png"),
    );
    // Only on change: reassigning src reloads the frame, which would throw
    // away a zoom the reader is in the middle of.
    if (state.frameSrc !== svg) {
      const frame = $("flamegraph-frame");
      // A flamegraph grows upward from its root, so the frame's own top-left
      // is the *tips* of the deepest stacks — a wall of slivers. Land on the
      // base, where the wide frames that account for the time are, and let
      // the reader scroll up into the detail.
      frame.addEventListener(
        "load",
        () => {
          try {
            const doc = frame.contentDocument;
            doc.defaultView.scrollTo(0, doc.documentElement.scrollHeight);
          } catch {
            // Nothing to do if the frame won't hand over its document; the
            // profile is still there to scroll by hand.
          }
        },
        { once: true },
      );
      frame.src = svg;
      state.frameSrc = svg;
    }

    // A flamegraph is normalised to its own total, so it can only say how the
    // time was divided. These say how much there was to divide.
    const p = view.profile ?? {};
    const stats = $("flamegraph-stats");
    stats.replaceChildren();
    const entries = [
      ["samples", p.samples],
      ["window", p.seconds != null ? `${p.seconds.toFixed(1)}s` : null],
      ["cores busy", p.cores_busy != null ? p.cores_busy.toFixed(2) : null],
      ["cores working", p.cores_working != null ? p.cores_working.toFixed(2) : null],
      ["GPU wait", p.gpu_wait_pct != null ? `${p.gpu_wait_pct.toFixed(1)}%` : null],
      ["pool idle", p.pool_idle_pct != null ? `${p.pool_idle_pct.toFixed(1)}%` : null],
      ["/proc says", p.cores_from_proc != null ? `${p.cores_from_proc.toFixed(2)} cores` : null],
    ];
    for (const [key, value] of entries) {
      if (value == null) continue;
      const span = document.createElement("span");
      span.append(`${key} `);
      const b = document.createElement("b");
      b.textContent = String(value);
      span.appendChild(b);
      stats.appendChild(span);
    }
  }

  function renderChart(view) {
    const artifacts = view.artifacts ?? [];
    const has = artifacts.includes("chart.svg");
    $("chart-section").hidden = !has;
    if (!has) return;
    $("chart-img").src = artifactUrl(view.id, "chart.svg");
    saveTarget($("chart-svg-dl"), view, "chart.svg", true);
    saveTarget($("chart-png-dl"), view, "chart.png", artifacts.includes("chart.png"));
  }

  function artifactUrl(id, name) {
    return `/api/runs/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(name)}`;
  }

  // Point a save control at one of this run's artifacts, and name the file it
  // writes after the run — two arms of an A/B saved from two tabs are both
  // "flamegraph.svg" otherwise, and the browser resolves that by appending
  // (1), which says nothing about which arm it came from.
  function saveTarget(el, view, name, present) {
    el.href = artifactUrl(view.id, name);
    el.download = `orangu-bench-${view.id}-${name}`;
    el.hidden = !present;
  }

  // ---- past runs -----------------------------------------------------

  $("history-btn").addEventListener("click", () => {
    const panel = $("history-panel");
    panel.hidden = !panel.hidden;
    $("history-btn").setAttribute("aria-expanded", String(!panel.hidden));
    if (!panel.hidden) loadHistory();
  });

  document.addEventListener("click", (event) => {
    const panel = $("history-panel");
    if (panel.hidden) return;
    if (panel.contains(event.target) || $("history-btn").contains(event.target)) return;
    panel.hidden = true;
    $("history-btn").setAttribute("aria-expanded", "false");
  });

  async function loadHistory() {
    let runs = [];
    try {
      const res = await fetch("/api/runs", { cache: "no-store" });
      runs = (await res.json()).runs ?? [];
    } catch {
      return;
    }
    state.runs = runs;
    // The comparison menu is built from this list, so it is redrawn here
    // rather than waiting for the next poll — which, on a finished run, never
    // comes. Redrawn from the view already in hand, never by re-polling:
    // render() calls loadHistory(), and calling back into it would loop.
    if (state.view) renderCompare(state.view);
    const list = $("history-list");
    list.replaceChildren();
    // Nothing to clear when there is nothing kept.
    $("history-footer").hidden = runs.length === 0;
    if (!runs.length) {
      const empty = document.createElement("div");
      empty.className = "history-empty";
      empty.textContent = "No runs yet.";
      list.appendChild(empty);
      return;
    }
    for (const run of runs) {
      list.appendChild(historyRow(run));
    }
  }

  // **Clear all**: every kept run, gone. A run still measuring is not one of
  // them — the server keeps it and says so, and this reports that rather than
  // leaving a list that looks like it failed to empty.
  $("history-clear-btn").addEventListener("click", async () => {
    if (!window.confirm("Delete every kept run?\n\nThis cannot be undone.")) return;
    let body = {};
    try {
      const res = await fetch("/api/runs", { method: "DELETE" });
      body = await res.json();
    } catch (e) {
      showError(`could not reach the console: ${e}`);
      return;
    }
    if (body.kept) {
      // The one run that survived is the one still going, so follow it —
      // anything else would hide a measurement in flight.
      attach(body.kept);
      loadHistory();
    } else {
      newRun();
    }
  });

  function historyRow(run) {
    const row = document.createElement("div");
    row.className = `history-row${run.id === state.id ? " current" : ""}`;
    row.addEventListener("click", () => {
      $("history-panel").hidden = true;
      $("history-btn").setAttribute("aria-expanded", "false");
      applySpec(run.spec);
      attach(run.id);
    });

    const main = document.createElement("div");
    main.className = "history-main";
    const title = document.createElement("div");
    title.className = "history-title";
    title.textContent = modeName(run.spec?.mode);
    const sub = document.createElement("div");
    sub.className = "history-sub";
    // The URL on the second line, not the first: two arms of an A/B are the
    // same measurement against different servers, so the part that tells them
    // apart must not be the part an over-long title truncates away.
    const when = run.started ? new Date(run.started * 1000).toLocaleString() : "";
    const secs = run.seconds ? ` · ${run.seconds.toFixed(1)}s` : "";
    sub.textContent = `${run.spec?.url ?? ""} · ${when}${secs}`;
    main.append(title, sub);

    const badge = document.createElement("span");
    badge.className = `badge ${run.status}`;
    badge.textContent = run.status;

    const del = document.createElement("button");
    del.className = "icon-btn subtle-btn";
    del.type = "button";
    del.title = "Delete this run";
    del.setAttribute("aria-label", "Delete this run");
    del.textContent = "✕";
    del.addEventListener("click", async (event) => {
      event.stopPropagation();
      await fetch(`/api/runs/${encodeURIComponent(run.id)}`, { method: "DELETE" });
      // Deleting the run on screen leaves the same empty pane **New** does,
      // through the same path — including forgetting it, so a reload does not
      // reopen a run that no longer exists.
      if (run.id === state.id) {
        newRun();
      } else {
        loadHistory();
      }
    });

    row.append(main, badge, del);
    return row;
  }

  // ---- startup -------------------------------------------------------

  (async function start() {
    paintThemeToggle();
    let defaults = null;
    try {
      const res = await fetch("/api/defaults", { cache: "no-store" });
      defaults = await res.json();
    } catch {
      showError("could not reach the console");
    }
    state.capabilities = {
      have_perf: defaults?.have_perf,
      have_rsvg: defaults?.have_rsvg,
    };
    state.presets = defaults?.presets ?? [];

    // The last run's definition, so a second A/B arm is one field away from
    // the first. Falls back to the CLI's own defaults.
    let saved = null;
    try {
      saved = JSON.parse(localStorage.getItem(SPEC_KEY) ?? "null");
    } catch {
      saved = null;
    }
    applySpec(saved ?? defaults?.spec ?? {});
    // After the spec, so the preset's own values win where it has them — it
    // is the thing that was selected, and the saved spec is only what that
    // selection last produced.
    paintPresets(localStorage.getItem(PRESET_KEY) ?? "");
    paintFlamegraph();

    // Whatever this console was last doing, in priority order: a run still
    // going (a reload, or a second tab — a measurement in flight must never
    // be hidden), then the run this browser was last reading, then the most
    // recent result. A reload two seconds after a twenty-minute sweep must
    // not land on an empty page with the answer one menu away.
    //
    // Nothing at all after **New**, which forgets the run on purpose.
    await loadHistory();
    // `null` — never chose one, so show the newest. `""` — **New** was
    // pressed, so show nothing; that is the whole point of the button, and it
    // has to survive a reload. An id whose run has since been deleted falls
    // back to the newest rather than to an empty page.
    const remembered = localStorage.getItem(RUN_KEY);
    let target = state.runs.find((r) => r.status === "running");
    if (!target && remembered !== "") {
      target = state.runs.find((r) => r.id === remembered) ?? state.runs[0];
    }
    if (target) attach(target.id);
  })();
})();
