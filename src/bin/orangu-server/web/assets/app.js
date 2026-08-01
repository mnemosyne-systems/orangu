(() => {
  "use strict";

  const transcript = document.getElementById("transcript");
  const input = document.getElementById("input");
  const composer = document.getElementById("composer");
  const sendBtn = document.getElementById("send-btn");
  const reloadBtn = document.getElementById("reload-btn");
  const newChatBtn = document.getElementById("new-chat-btn");
  const historyBtn = document.getElementById("history-btn");
  const historyPanel = document.getElementById("history-panel");
  const historyList = document.getElementById("history-list");
  const historyFooter = document.getElementById("history-footer");
  const historyClearBtn = document.getElementById("history-clear-btn");
  const themeToggleBtn = document.getElementById("theme-toggle-btn");
  const attachBtn = document.getElementById("attach-btn");
  const attachMenu = document.getElementById("attach-menu");
  const attachmentsEl = document.getElementById("attachments");
  const attachInputs = {
    document: document.getElementById("attach-input-document"),
    file: document.getElementById("attach-input-file"),
  };

  const state = { sessionId: null, busy: false, abortController: null };

  // Files staged for the next message. Each entry is
  // {name, mime, size, data} where `data` is base64 of the raw bytes — the
  // server decodes it and extracts text (documents) or notes it as a
  // reference (binaries), since the engine is text-only.
  let pendingAttachments = [];
  const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024;

  // Swapped into #send-btn by setBusy() below — Send while idle, a plain
  // "X" while a reply is streaming so the same button can cancel it.
  const SEND_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;
  const STOP_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
  // Shown in each assistant message's footer, next to the generation time
  // — triggers a raw-Markdown download of that answer.
  const SAVE_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;

  const THEME_KEY = "orangu-theme";

  function effectiveTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function renderThemeToggle() {
    const dark = effectiveTheme() === "dark";
    const label = dark ? "Switch to light mode" : "Switch to dark mode";
    themeToggleBtn.textContent = dark ? "☀️" : "🌙";
    themeToggleBtn.setAttribute("aria-label", label);
    themeToggleBtn.setAttribute("title", label);
  }

  themeToggleBtn.addEventListener("click", () => {
    localStorage.setItem(THEME_KEY, effectiveTheme() === "dark" ? "light" : "dark");
    document.documentElement.setAttribute("data-theme", effectiveTheme());
    renderThemeToggle();
  });

  renderThemeToggle();

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function addMessage(role, text, attachments) {
    const el = document.createElement("div");
    el.className = `message ${role}`;
    el.textContent = text;
    appendAttachmentChips(el, attachments);
    transcript.appendChild(el);
    transcript.scrollTop = transcript.scrollHeight;
    return el;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  // One attachment chip: name + size, and (staged only) a remove button.
  function makeAttachmentChip(att, onRemove) {
    const chip = document.createElement("span");
    chip.className = "attachment-chip";

    const name = document.createElement("span");
    name.className = "chip-name";
    name.textContent = att.name;
    name.title = att.mime ? `${att.name} (${att.mime})` : att.name;

    const size = document.createElement("span");
    size.className = "chip-size";
    size.textContent = formatSize(att.size);

    chip.append(name, size);

    if (onRemove) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "chip-remove";
      remove.setAttribute("aria-label", `Remove ${att.name}`);
      remove.textContent = "×";
      remove.addEventListener("click", onRemove);
      chip.appendChild(remove);
    }
    return chip;
  }

  // Display-only attachments under a sent (or reloaded) user message.
  // Replaces any previous strip, so the `attachments` SSE event can swap
  // the plain chips this renders at send time for the server's richer
  // views once the file has actually been read.
  function appendAttachmentChips(el, attachments) {
    const previous = el.querySelector(":scope > .message-attachments");
    if (previous) previous.remove();
    if (!attachments || attachments.length === 0) return;
    const container = document.createElement("div");
    container.className = "message-attachments";
    for (const att of attachments) {
      container.appendChild(makeAttachmentEntry(att));
    }
    el.appendChild(container);
  }

  // One sent attachment. When the server could read the file, the chip
  // becomes the summary of a collapsed panel holding what it read —
  // diagrams as pictures, then the extracted text. When it couldn't (a
  // binary or unrecognised type: no MIME we can do anything with, nothing
  // extracted) there is nothing behind an expand control, so the chip stays
  // a plain chip and no control is offered.
  //
  // Collapsed by default: the point of a chip is that a message stays
  // readable regardless of how much was attached to it.
  function makeAttachmentEntry(att) {
    const hasDiagrams = att.diagrams && att.diagrams.length > 0;
    const hasText = typeof att.text === "string" && att.text.length > 0;
    if (!hasDiagrams && !hasText) return makeAttachmentChip(att, null);

    const details = document.createElement("details");
    details.className = "attachment-details";

    const summary = document.createElement("summary");
    summary.appendChild(makeAttachmentChip(att, null));
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "attachment-body";
    if (hasDiagrams) {
      for (const diagram of att.diagrams) {
        body.appendChild(makeDiagram(diagram, att.name));
      }
      if (att.diagrams_capped) {
        const note = document.createElement("p");
        note.className = "diagram-note";
        note.textContent = `Showing the first ${att.diagrams.length} diagrams in ${att.name}.`;
        body.appendChild(note);
      }
    }
    if (hasText) {
      // Exactly the text that went to the model. `textContent`, so an
      // uploaded file's contents can never be markup here.
      const pre = document.createElement("pre");
      pre.className = "attachment-text";
      const code = document.createElement("code");
      code.textContent = att.text;
      pre.appendChild(code);
      body.appendChild(pre);
    }
    details.appendChild(body);
    return details;
  }

  // The same markup `render.rs` emits for a diagram in a reply, so one set
  // of CSS rules covers both: an <img> per theme plus the collapsed source.
  function makeDiagram(diagram, fileName) {
    const figure = document.createElement("figure");
    figure.className = "mermaid-diagram";

    for (const theme of ["light", "dark"]) {
      const img = document.createElement("img");
      img.className = `mermaid-${theme}`;
      img.src = diagram[theme];
      img.alt = fileName ? `Mermaid diagram from ${fileName}` : "Mermaid diagram";
      // Natural size, so a wide diagram stays legible and the figure
      // scrolls instead of scaling it down. See `web::mermaid`.
      if (diagram.width > 0) {
        img.width = Math.round(diagram.width);
        img.height = Math.round(diagram.height);
      }
      figure.appendChild(img);
    }

    // Same download control `render.rs` puts on a diagram in a reply: the
    // picture is scaled to the message, so this is how the full-resolution
    // original gets out. One per theme, so the saved file matches what's on
    // screen. Plain anchors onto the `data:` URI already in the <img>.
    const actions = document.createElement("div");
    actions.className = "diagram-actions";
    for (const theme of ["light", "dark"]) {
      const link = document.createElement("a");
      link.className = `diagram-dl diagram-dl-${theme}`;
      link.href = diagram[theme];
      link.download = fileName ? `${fileName}.svg` : "orangu-diagram.svg";
      link.title = "Download SVG";
      link.setAttribute("aria-label", "Download diagram as SVG");
      link.innerHTML = SAVE_ICON;
      actions.appendChild(link);
    }
    figure.appendChild(actions);

    const details = document.createElement("details");
    details.className = "mermaid-source";
    const summary = document.createElement("summary");
    summary.textContent = "Diagram source";
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = diagram.source;
    pre.appendChild(code);
    details.append(summary, pre);
    figure.appendChild(details);

    return figure;
  }

  // Re-render the staged-attachment strip above the input from
  // `pendingAttachments`, each chip removable.
  function renderPendingAttachments() {
    attachmentsEl.innerHTML = "";
    for (const att of pendingAttachments) {
      attachmentsEl.appendChild(
        makeAttachmentChip(att, () => {
          pendingAttachments = pendingAttachments.filter((a) => a !== att);
          renderPendingAttachments();
        }),
      );
    }
    attachmentsEl.hidden = pendingAttachments.length === 0;
  }

  function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        // readAsDataURL gives "data:<mime>;base64,XXXX" — keep just XXXX.
        const result = String(reader.result);
        const comma = result.indexOf(",");
        resolve(comma >= 0 ? result.slice(comma + 1) : result);
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }

  async function stageFiles(fileList) {
    for (const file of Array.from(fileList || [])) {
      if (file.size > MAX_ATTACHMENT_BYTES) {
        window.alert(`"${file.name}" is larger than 25 MB and was skipped.`);
        continue;
      }
      try {
        const data = await readFileAsBase64(file);
        pendingAttachments.push({ name: file.name, mime: file.type || "", size: file.size, data });
      } catch (err) {
        console.error("failed to read attachment", file.name, err);
      }
    }
    renderPendingAttachments();
  }

  function addRenderedMessage(role, html) {
    const el = document.createElement("div");
    el.className = `message ${role}`;
    el.innerHTML = html;
    transcript.appendChild(el);
    transcript.scrollTop = transcript.scrollHeight;
    return el;
  }

  // Shortest colon-separated D:H:M:S form that fits — leading all-zero
  // units are dropped entirely rather than shown as "0:", so a typical
  // few-second generation reads as "12s", not "0:00:00:12".
  function formatDuration(ms) {
    let totalSeconds = Math.round(ms / 1000);
    const days = Math.floor(totalSeconds / 86400);
    totalSeconds -= days * 86400;
    const hours = Math.floor(totalSeconds / 3600);
    totalSeconds -= hours * 3600;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds - minutes * 60;
    const pad = (n) => String(n).padStart(2, "0");

    if (days > 0) return `${days}:${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
    if (hours > 0) return `${hours}:${pad(minutes)}:${pad(seconds)}`;
    if (minutes > 0) return `${minutes}:${pad(seconds)}`;
    return `${seconds}s`;
  }

  // Triggers the browser's native download ("Save As", depending on the
  // user's download-prompt setting) for `content` as a standalone
  // `.md` file — a Blob + object URL fed through a throwaway anchor's
  // `download` attribute, the standard way to save client-side-only
  // content without a server round trip.
  function downloadMarkdown(content) {
    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date()
      .toISOString()
      .replace(/[:T]/g, "-")
      .slice(0, 19);
    a.href = url;
    a.download = `orangu-answer-${stamp}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // Same download mechanism as `downloadMarkdown`, plain text instead of
  // markdown — used for the error-bubble debug report below.
  function downloadTextFile(content, filenamePrefix) {
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date()
      .toISOString()
      .replace(/[:T]/g, "-")
      .slice(0, 19);
    a.href = url;
    a.download = `${filenamePrefix}-${stamp}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // The whole visible transcript, read back out of the DOM rather than
  // kept in a parallel JS structure — `state` deliberately holds nothing
  // but `sessionId`/`busy`/`abortController` (see its own declaration), and
  // the DOM is the one place a turn that *failed* (never persisted to the
  // session file — `sessions::append_turn` only runs on success, see
  // `web/mod.rs`) still exists at all, alongside every earlier, actually-
  // persisted turn. Good enough for a debug report: plain rendered text,
  // not the raw markdown/HTML.
  function collectConversationText() {
    const parts = [];
    for (const child of transcript.children) {
      const role = child.classList.contains("user")
        ? "user"
        : child.classList.contains("assistant")
          ? "assistant"
          : child.classList.contains("error")
            ? "error"
            : "unknown";
      const text = (child.innerText ?? child.textContent ?? "").trim();
      parts.push(`[${role}]\n${text}`);
    }
    return parts.length > 0 ? parts.join("\n\n") : "(empty)";
  }

  // Everything a bug report needs beyond "it broke": the server's own
  // `orangu-server system` report plus model/backend identity (`/api/
  // system-report`, fetched fresh so it reflects VRAM/RAM *right now*, not
  // whatever it was at server startup), the full visible conversation, and
  // the error's own detail — for a panic, `detail` is already the real
  // message plus a captured backtrace (`panic_capture`, `engine::
  // generate::Engine::generate`), not just the generic "task panicked"
  // note `tokio::task::JoinError`'s own `Display` would otherwise give.
  async function buildDebugReport(detail) {
    let systemReport = "(failed to fetch: /api/system-report unreachable)";
    try {
      const res = await fetch("/api/system-report", { cache: "no-store" });
      systemReport = res.ok
        ? await res.text()
        : `(failed to fetch: HTTP ${res.status})`;
    } catch (err) {
      systemReport = `(failed to fetch: ${err})`;
    }
    const detailText =
      detail instanceof Error ? detail.stack || detail.message : String(detail ?? "");

    return [
      "orangu-server web UI debug report",
      `Generated: ${new Date().toISOString()}`,
      "",
      "== System ==",
      systemReport.trimEnd(),
      "",
      "== Conversation ==",
      collectConversationText(),
      "",
      "== Error detail ==",
      detailText,
    ].join("\n");
  }

  // Mirrors `addTimingFooter`'s own shape (a `.gen-time` bar with a save
  // button) but for an error bubble: no generation time to show, and the
  // save button assembles/downloads the debug report above instead of a
  // single answer's raw markdown.
  function addErrorFooter(assistantEl, detail) {
    const footer = document.createElement("div");
    footer.className = "gen-time";

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "save-md-btn";
    saveBtn.innerHTML = SAVE_ICON;
    saveBtn.setAttribute("aria-label", "Save debug report");
    saveBtn.setAttribute("title", "Save debug report");
    saveBtn.addEventListener("click", () => {
      buildDebugReport(detail)
        .then((report) => downloadTextFile(report, "orangu-debug-report"))
        .catch((err) => console.error("failed to build debug report:", err));
    });
    footer.appendChild(saveBtn);

    assistantEl.appendChild(footer);
  }

  // Appended once generation finishes (streamed replies only know their
  // own time and raw text at the "done" event; history reloads know both
  // right away from the loaded session) — deliberately its own element
  // rather than baked into the rendered markdown, so it survives
  // `assistantEl.innerHTML = payload.html` reassignments during streaming
  // and never gets treated as message content (copy/paste, markdown
  // re-render, ...).
  function addTimingFooter(assistantEl, ms, rawContent, tpsText) {
    if (ms == null) return;
    const footer = document.createElement("div");
    footer.className = "gen-time";

    // Left-aligned tokens-per-second, kept from the live readout so the
    // final footer shows the same figure the counter settled on (only
    // freshly streamed replies have it — history reloads pass nothing).
    if (tpsText) {
      const rate = document.createElement("span");
      rate.className = "gen-tps";
      rate.textContent = tpsText;
      footer.appendChild(rate);
    }

    const time = document.createElement("span");
    time.textContent = formatDuration(ms);
    footer.appendChild(time);

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "save-md-btn";
    saveBtn.innerHTML = SAVE_ICON;
    saveBtn.setAttribute("aria-label", "Save answer as Markdown");
    saveBtn.setAttribute("title", "Save answer as Markdown");
    saveBtn.addEventListener("click", () => downloadMarkdown(rawContent ?? ""));
    footer.appendChild(saveBtn);

    assistantEl.appendChild(footer);
  }

  // While a code block is still filling up during streaming, keep it
  // scrolled to its latest line (like `tail -f`) instead of leaving it
  // pinned to the top — the horizontal/vertical scrollbars (`.message
  // pre` in app.css) stay available throughout for manual scrolling.
  function pinCodeBlocksToLatest(el) {
    for (const pre of el.querySelectorAll("pre")) {
      pre.scrollTop = pre.scrollHeight;
    }
  }

  // Typesets every `<span/div class="katex-source" data-tex="...">`
  // placeholder `render.rs` emits for `$...$`/`$$...$$` math — in place,
  // via KaTeX's own `render()` (bundled locally, see index.html; no CDN,
  // this has to work fully offline). `data-tex` round-trips through the
  // DOM already HTML-entity-decoded, so no unescaping is needed here.
  // Malformed TeX (`throwOnError: false`) just leaves the element's
  // existing escaped-source text in place instead of blanking it.
  function renderMathIn(el) {
    if (typeof katex === "undefined") return;
    for (const node of el.querySelectorAll(".katex-source")) {
      try {
        katex.render(node.dataset.tex, node, {
          throwOnError: false,
          displayMode: node.classList.contains("katex-block"),
        });
      } catch (err) {
        console.error("katex render failed:", err);
      }
    }
  }

  // Puts the diagrams from a turn's attachments into that turn's answer.
  //
  // Asked to render an attached diagram, models reliably answer "Here is
  // the rendered content" and then *describe* it in prose — the explanation
  // is right, but no Mermaid comes back, so there is nothing in the reply
  // for the renderer to draw. The picture the reader asked for exists; it
  // just came from the file rather than the model.
  //
  // Two rules keep this honest. It only fires when the answer contains no
  // diagram of its own, so a model that does emit Mermaid is never
  // second-guessed or duplicated. And each figure is captioned with the
  // file it came from, so nothing here reads as something the model drew —
  // the saved message text stays exactly what the model wrote, which is
  // also what the Save-as-Markdown button and the next turn's context see.
  function appendAttachedDiagramsToAnswer(assistantEl, attachments) {
    if (!attachments || attachments.length === 0) return;
    if (assistantEl.querySelector(".mermaid-diagram")) return;

    for (const att of attachments) {
      for (const diagram of att.diagrams || []) {
        const figure = makeDiagram(diagram, att.name);
        const caption = document.createElement("figcaption");
        caption.className = "diagram-provenance";
        caption.textContent = `From ${att.name}`;
        figure.appendChild(caption);
        assistantEl.appendChild(figure);
      }
    }
  }

  // sendBtn stays enabled throughout a request — while idle it submits the
  // form, while busy its click handler (below) cancels the in-flight
  // request instead, so it can't be disabled the way `input` is.
  function setBusy(busy) {
    state.busy = busy;
    input.disabled = busy;
    attachBtn.disabled = busy;
    sendBtn.classList.toggle("stop", busy);
    sendBtn.innerHTML = busy ? STOP_ICON : SEND_ICON;
    sendBtn.setAttribute("aria-label", busy ? "Stop" : "Send");
    sendBtn.setAttribute("title", busy ? "Stop" : "Send");
  }

  // Aborting the fetch closes the SSE connection, which drops the server's
  // receiver on the generation channel — the engine notices the next time
  // it tries to send a token and stops decoding right there (cooperative,
  // not instant, but no explicit server-side cancel endpoint is needed).
  function stopGeneration() {
    if (state.abortController) {
      state.abortController.abort();
    }
  }

  async function createSession() {
    const res = await fetch("/api/sessions", { method: "POST" });
    if (!res.ok) throw new Error(`failed to create session (${res.status})`);
    return res.json();
  }

  // Swaps the empty transcript in without touching History's own
  // visibility — the delete paths below need exactly this and must not
  // close the panel out from under someone tidying up several chats.
  async function startFreshSession() {
    const session = await createSession();
    state.sessionId = session.id;
    localStorage.setItem("orangu-session-id", session.id);
    transcript.innerHTML = "";
  }

  async function newChat() {
    await startFreshSession();
    hideHistory();
  }

  async function loadSession(id) {
    const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error(`failed to load session (${res.status})`);
    const session = await res.json();
    state.sessionId = session.id;
    localStorage.setItem("orangu-session-id", session.id);
    transcript.innerHTML = "";
    // Diagrams from the attachments on the turn being answered, so a
    // reloaded answer carries the same picture a live one did.
    let pendingTurnDiagrams = [];
    for (const message of session.messages) {
      if (message.role === "assistant") {
        const el = addRenderedMessage("assistant", message.html || escapeHtml(message.content));
        renderMathIn(el);
        appendAttachedDiagramsToAnswer(el, pendingTurnDiagrams);
        addTimingFooter(el, message.generation_ms, message.content);
        pendingTurnDiagrams = [];
      } else {
        addMessage(message.role, message.content, message.attachments);
        pendingTurnDiagrams = message.attachments || [];
      }
    }
    hideHistory();
  }

  function formatDate(unixSeconds) {
    return new Date(unixSeconds * 1000).toLocaleString();
  }

  function historyTitle(session) {
    return session.title || "New chat";
  }

  async function refreshHistory() {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const sessions = await res.json();
    // Nothing to clear is the only reason the footer ever hides — deleting
    // sessions is unconditional, unlike the model manager's own Delete.
    historyFooter.hidden = sessions.length === 0;
    historyList.innerHTML = "";
    if (sessions.length === 0) {
      const empty = document.createElement("div");
      empty.className = "history-empty";
      empty.textContent = "No previous chats yet.";
      historyList.appendChild(empty);
      return;
    }
    for (const session of sessions) {
      const item = document.createElement("div");
      item.className = "history-item";
      const text = document.createElement("div");
      text.className = "history-item-text";
      const title = document.createElement("div");
      title.className = "history-title";
      title.textContent = historyTitle(session);
      const date = document.createElement("div");
      date.className = "history-date";
      date.textContent = formatDate(session.updated_at);
      text.appendChild(title);
      text.appendChild(date);
      item.appendChild(text);
      item.addEventListener("click", () => {
        loadSession(session.id).catch((err) => console.error(err));
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "history-delete";
      remove.innerHTML = ICON.close;
      remove.title = `Delete "${historyTitle(session)}"`;
      remove.setAttribute("aria-label", remove.title);
      remove.addEventListener("click", (event) => {
        // The row itself opens the chat — without this, deleting one would
        // load it on the way out.
        event.stopPropagation();
        deleteSession(session).catch((err) => console.error(err));
      });
      item.appendChild(remove);
      historyList.appendChild(item);
    }
  }

  // Deleting the session a reply is still streaming into would otherwise
  // see the server write the finished turn back out on "done" —
  // `save_session` recreates the directory — and the chat someone just
  // deleted would reappear a few seconds later. Stopping first is exactly
  // what the Stop button does: the turn never completes, so nothing is
  // persisted.
  function stopIfGeneratingInto(id) {
    if (state.busy && id === state.sessionId) stopGeneration();
  }

  // Both delete paths keep the panel open and re-list afterwards, so the
  // row disappearing is the confirmation. If what went was the chat
  // currently on screen, a fresh empty session takes its place — the
  // transcript would otherwise go on showing a conversation that no longer
  // exists, and the next message sent would 404 against its id.
  async function deleteSession(session) {
    if (
      !window.confirm(`Delete "${historyTitle(session)}"?\n\nThis cannot be undone.`)
    ) {
      return;
    }
    stopIfGeneratingInto(session.id);
    const res = await fetch(`/api/sessions/${encodeURIComponent(session.id)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      console.error("failed to delete session", await res.text());
      return;
    }
    if (session.id === state.sessionId) await startFreshSession();
    await refreshHistory();
  }

  async function clearHistory() {
    if (!window.confirm("Delete every saved chat?\n\nThis cannot be undone.")) return;
    stopIfGeneratingInto(state.sessionId);
    const res = await fetch("/api/sessions", { method: "DELETE" });
    if (!res.ok) {
      console.error("failed to clear history", await res.text());
      return;
    }
    // Unconditionally, not just when the current chat was in the list: an
    // unsent new chat isn't listed but its directory went with the rest.
    await startFreshSession();
    await refreshHistory();
  }

  function showHistory() {
    refreshHistory().catch((err) => console.error(err));
    historyPanel.hidden = false;
    historyBtn.setAttribute("aria-expanded", "true");
  }

  function hideHistory() {
    historyPanel.hidden = true;
    historyBtn.setAttribute("aria-expanded", "false");
  }

  // Shown in the chat on any failure — the real detail always goes to the
  // browser console (console.error) instead, for whoever's actually
  // debugging it; a chat bubble full of a stack trace or a template-
  // rendering error isn't useful to someone just trying to send a message.
  // The footer's Save button (below) is where that detail actually goes
  // for someone who *does* want it, bundled into a full debug report.
  const FAILURE_MESSAGE = "🦧";

  function showFailure(assistantEl, consoleLabel, detail) {
    console.error(consoleLabel, detail);
    assistantEl.className = "message error";
    assistantEl.textContent = FAILURE_MESSAGE;
    addErrorFooter(assistantEl, detail);
  }

  async function sendMessage(text, attachments) {
    attachments = attachments || [];
    if (!state.sessionId) {
      await newChat();
    }
    // Kept so the "attachments" event below can hang diagrams off the user's
    // own message once the server has drawn them (the browser only has the
    // raw file bytes; extraction and rendering are server-side).
    const userEl = addMessage("user", text, attachments);
    const assistantEl = addMessage("assistant", "🤖");
    assistantEl.classList.add("pending");
    setBusy(true);
    const controller = new AbortController();
    state.abortController = controller;

    // Live tokens-per-second for this answer's footer. orangu-server emits
    // one SSE "token" event per generated token, so counting those events
    // is the token count with no extra server plumbing. The clock starts on
    // the first token (not on send, which would fold prompt-processing
    // latency into the rate) and that first token is excluded from the
    // count, so the figure is steady-state inter-token throughput.
    let tpsStarted = false;
    let tpsStartMs = 0;
    let tpsCount = 0;
    let liveFooter = null;
    // The server's view of this turn's uploads, as sent by the "attachments"
    // event — kept so the finished answer can carry their diagrams.
    let turnAttachments = [];
    const tpsText = () => {
      if (!tpsStarted || tpsCount === 0) return null;
      const elapsed = (performance.now() - tpsStartMs) / 1000;
      return elapsed > 0 ? `${(tpsCount / elapsed).toFixed(1)} t/s` : null;
    };

    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: text,
          attachments: attachments.map((a) => ({ name: a.name, mime: a.mime, data: a.data })),
        }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        const detail = await res.text().catch(() => "");
        throw new Error(`request failed (${res.status})${detail ? `: ${detail}` : ""}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let sseBuffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        sseBuffer += decoder.decode(value, { stream: true });
        const events = sseBuffer.split("\n\n");
        sseBuffer = events.pop() ?? "";
        for (const raw of events) {
          const line = raw.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const payload = JSON.parse(line.slice("data: ".length));
          if (payload.type === "attachments") {
            // Sent before the first token: the browser only ever had the raw
            // bytes, so this is the first time it learns what the server
            // read out of them. Swaps the plain chips rendered at send time
            // for expandable ones. The assistant bubble stays "pending" —
            // nothing has been generated yet — so this deliberately sits
            // above the type check below.
            turnAttachments = payload.attachments || [];
            appendAttachmentChips(userEl, turnAttachments);
            continue;
          }
          assistantEl.classList.remove("pending");
          if (payload.type === "token" || payload.type === "done") {
            assistantEl.innerHTML = payload.html;
            pinCodeBlocksToLatest(assistantEl);
            renderMathIn(assistantEl);
            if (payload.type === "token") {
              if (!tpsStarted) {
                tpsStarted = true;
                tpsStartMs = performance.now();
              } else {
                tpsCount += 1;
              }
              // `innerHTML = payload.html` above wipes the message's
              // children every token, so the live footer can't be attached
              // just once — build it lazily, keep the reference, and
              // re-append it after each re-render (same reason the final
              // footer waits for "done"). Left-aligned via `.gen-tps`.
              const text = tpsText();
              if (text) {
                if (!liveFooter) {
                  liveFooter = document.createElement("div");
                  liveFooter.className = "gen-time";
                  const rate = document.createElement("span");
                  rate.className = "gen-tps";
                  liveFooter.appendChild(rate);
                }
                liveFooter.firstChild.textContent = text;
                assistantEl.appendChild(liveFooter);
              }
            }
            if (payload.type === "done") {
              // Only once the answer is final: every token above reassigns
              // `innerHTML`, which would wipe anything appended earlier, and
              // an answer still mid-sentence hasn't yet had its chance to
              // produce a diagram of its own.
              appendAttachedDiagramsToAnswer(assistantEl, turnAttachments);
              if (payload.truncated) {
                const notice = document.createElement("p");
                notice.className = "truncated-notice";
                notice.textContent = "⚠️ Response was cut off at the token limit.";
                assistantEl.appendChild(notice);
              }
              addTimingFooter(assistantEl, payload.generation_ms, payload.content, tpsText());
            }
            transcript.scrollTop = transcript.scrollHeight;
          } else if (payload.type === "error") {
            showFailure(assistantEl, "orangu-server generation error:", payload.message);
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        // User-initiated stop, not a failure — leave whatever text already
        // streamed in place (marked as stopped) instead of showing the
        // failure bubble. If nothing had arrived yet, drop the placeholder.
        const hadContent = !assistantEl.classList.contains("pending");
        assistantEl.classList.remove("pending");
        if (hadContent) {
          const notice = document.createElement("p");
          notice.className = "truncated-notice";
          notice.textContent = "⏹️ Stopped.";
          assistantEl.appendChild(notice);
        } else {
          assistantEl.remove();
        }
      } else {
        showFailure(assistantEl, "orangu-server request failed:", err);
      }
    } finally {
      setBusy(false);
      state.abortController = null;
    }
  }

  // While busy, sendBtn is a Stop button: intercept its click before the
  // browser's default submit action fires, so it cancels instead of
  // re-submitting the (disabled, empty) composer.
  sendBtn.addEventListener("click", (event) => {
    if (state.busy) {
      event.preventDefault();
      stopGeneration();
    }
  });

  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.busy) return;
    const text = input.value.trim();
    if (!text && pendingAttachments.length === 0) return;
    const attachments = pendingAttachments;
    pendingAttachments = [];
    renderPendingAttachments();
    input.value = "";
    sendMessage(text, attachments).catch((err) => console.error(err));
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });

  newChatBtn.addEventListener("click", () => {
    newChat().catch((err) => console.error(err));
  });

  // Attach ("+") button: toggle the Document/File menu; each item opens the
  // matching hidden <input type=file>.
  function setAttachMenu(open) {
    attachMenu.hidden = !open;
    attachBtn.setAttribute("aria-expanded", open ? "true" : "false");
  }

  attachBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    setAttachMenu(attachMenu.hidden);
  });

  for (const item of attachMenu.querySelectorAll("button[data-attach]")) {
    item.addEventListener("click", () => {
      setAttachMenu(false);
      attachInputs[item.dataset.attach].click();
    });
  }

  for (const inputEl of Object.values(attachInputs)) {
    inputEl.addEventListener("change", () => {
      const files = inputEl.files;
      inputEl.value = ""; // allow re-picking the same file later
      stageFiles(files).catch((err) => console.error(err));
    });
  }

  // Close the menu on an outside click or Escape.
  document.addEventListener("click", (event) => {
    if (!attachMenu.hidden && event.target !== attachBtn && !attachMenu.contains(event.target)) {
      setAttachMenu(false);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setAttachMenu(false);
  });

  reloadBtn.addEventListener("click", () => {
    window.location.reload();
  });

  // The Reload button stays hidden (see index.html) until the running
  // server's assets no longer match what this page was loaded with —
  // otherwise there's nothing for it to fix.
  const ASSET_VERSION = window.__ORANGU_ASSET_VERSION__;
  const UPDATE_CHECK_INTERVAL_MS = 60000;

  async function checkForUpdate() {
    if (!reloadBtn.hidden) return;
    try {
      const res = await fetch("/api/asset-version", { cache: "no-store" });
      if (!res.ok) return;
      const { version } = await res.json();
      if (version && version !== ASSET_VERSION) {
        reloadBtn.hidden = false;
      }
    } catch {
      // Server unreachable right now — nothing to report.
    }
  }

  setInterval(() => checkForUpdate().catch((err) => console.error(err)), UPDATE_CHECK_INTERVAL_MS);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") checkForUpdate().catch((err) => console.error(err));
  });
  checkForUpdate().catch((err) => console.error(err));

  historyBtn.addEventListener("click", () => {
    if (historyPanel.hidden) {
      showHistory();
    } else {
      hideHistory();
    }
  });

  historyClearBtn.addEventListener("click", () => {
    clearHistory().catch((err) => console.error(err));
  });

  // ---------------------------------------------------------------- models

  // The model manager (see web/models.rs): `orangu-server list`'s own table
  // as the view — NR / MODEL / QUANT / SIZE / SUPPORTED, the same strings
  // the CLI prints, rendered server-side so the two cannot disagree — plus
  // Show and Delete per row and a download box above it. Every action is an
  // icon button whose label lives in its `title`/`aria-label`, matching the
  // rest of this UI's chrome.

  const modelsBtn = document.getElementById("models-btn");
  const modelsOverlay = document.getElementById("models-overlay");
  const modelsCloseBtn = document.getElementById("models-close-btn");
  const modelsReloadBtn = document.getElementById("models-reload-btn");
  const modelsDirEl = document.getElementById("models-dir");
  const modelsCurrentEl = document.getElementById("models-current");
  const modelsTableEl = document.getElementById("models-table");
  const modelsJobEl = document.getElementById("models-job");
  const modelsNoticeEl = document.getElementById("models-notice");
  const modelsDownloadForm = document.getElementById("models-download-form");
  const modelsDownloadInput = document.getElementById("models-download-input");
  const modelsMetadataEl = document.getElementById("models-metadata");
  const modelsMetadataTitle = document.getElementById("models-metadata-title");
  const modelsMetadataBody = document.getElementById("models-metadata-body");
  const modelsMetadataTensorsBtn = document.getElementById("models-metadata-tensors-btn");
  const modelsMetadataFullBtn = document.getElementById("models-metadata-full-btn");
  const modelsMetadataSaveBtn = document.getElementById("models-metadata-save-btn");
  const modelsMetadataCloseBtn = document.getElementById("models-metadata-close-btn");

  const ICON = {
    // Load — a play triangle: start serving this one.
    load: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="6 3 20 12 6 21 6 3"/></svg>`,
    // The loaded row's marker, in place of its Load button.
    loaded: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
    // Show — a document with lines on it, for "this file's metadata".
    show: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="14" y2="17"/></svg>`,
    trash: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`,
    close: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
  };

  // How often the panel re-reads /api/models while it's open. A download's
  // progress is the only thing that moves on its own, and it moves in
  // megabytes, so a second is plenty. The poll deliberately does *not* pass
  // `rescan` — re-reading the models directory opens every GGUF header under
  // it (seconds, on a directory holding a few dozen models), and nothing on
  // disk changes on its own. A rescan is asked for exactly where the
  // directory can have changed: when the panel opens, after an action, and
  // from the Rescan button.
  const MODELS_POLL_MS = 1000;

  const modelsState = {
    open: false,
    timer: null,
    busy: false,
    // Row numbers the Hub says are behind their repo, applied to the next
    // listing — `list`'s own `(Refresh)` marker.
    behind: new Set(),
    // Which model's metadata the viewer is showing, and how — kept so the
    // tensors/full toggles can re-fetch the same target.
    metadata: { model: null, title: "", tensors: false, full: false, text: "" },
    // What the table was last built from. The poll runs once a second and
    // the table almost never changes between two of them — rebuilding it
    // anyway would drop hover state and any tooltip the pointer is resting
    // on, once a second, for nothing.
    tableSignature: null,
    // Set across a Load: the poll below expects failures then, and must not
    // report them as the panel breaking.
    handingOver: false,
  };

  function formatBytes(bytes) {
    if (bytes == null) return "";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`;
  }

  function formatEta(seconds) {
    if (seconds == null) return "";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 1) return "<1m";
    if (minutes < 60) return `${minutes}m`;
    return `${Math.floor(minutes / 60)}h:${String(minutes % 60).padStart(2, "0")}m`;
  }

  function iconButton(icon, label, onClick, extraClass) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `icon-btn row-btn${extraClass ? ` ${extraClass}` : ""}`;
    btn.innerHTML = icon;
    btn.setAttribute("aria-label", label);
    btn.title = label;
    if (onClick) btn.addEventListener("click", onClick);
    return btn;
  }

  // A transient line above the table: the result of the last action, or the
  // reason it was refused. Errors carry the server's own message verbatim —
  // "this model is currently loaded", "not enough free space", and the like
  // are exactly what the user needs to read.
  function modelsNotice(message, kind) {
    modelsNoticeEl.textContent = message;
    modelsNoticeEl.className = kind === "error" ? "models-notice error" : "models-notice";
    modelsNoticeEl.hidden = !message;
  }

  // Every mutating action goes through here: it serializes them (one at a
  // time, so a delete can't race the load that would have made it legal),
  // shows the outcome, and re-reads the listing afterwards either way.
  async function modelsAction(label, request) {
    if (modelsState.busy) return;
    modelsState.busy = true;
    modelsNotice(`${label}…`);
    modelsOverlay.classList.add("busy");
    try {
      const res = await request();
      const text = await res.text();
      if (!res.ok) {
        modelsNotice(text.trim() || `failed (${res.status})`, "error");
      } else {
        let message = "";
        try {
          message = JSON.parse(text).message || "";
        } catch {
          message = "";
        }
        modelsNotice(message);
      }
    } catch (err) {
      modelsNotice(String(err), "error");
    } finally {
      modelsState.busy = false;
      modelsOverlay.classList.remove("busy");
      // A rescan, not a cached read: an action is exactly what changes the
      // models directory.
      await refreshModels(true);
    }
  }

  // Which of the listed rows is actually serving requests, and on what.
  function renderCurrent(current, loading) {
    modelsCurrentEl.innerHTML = "";
    const name = document.createElement("div");
    name.className = "models-current-name";
    name.textContent = current.display;
    const detail = document.createElement("div");
    detail.className = "models-subtle";
    detail.textContent =
      `${current.architecture} · ${current.backend} · ${current.n_layer} layers · ` +
      `${current.n_ctx} ctx · ${current.role} · ${current.slots} slot(s)` +
      // A bundled server's model is inside the executable rather than in the
      // directory below, so it has no row to be marked "loaded" on and no
      // Delete button anywhere — which needs saying, or the listing reads as
      // if none of these models were serving anything.
      (current.bundled ? " · bundled" : "");
    detail.title = current.bundled
      ? `embedded in ${current.path}`
      : current.path;
    modelsCurrentEl.append(name, detail);
    if (loading) {
      const notice = document.createElement("div");
      notice.className = "models-loading";
      notice.textContent = `Loading ${loading}… the server is restarting itself on it.`;
      modelsCurrentEl.appendChild(notice);
    }
  }

  function renderJob(job) {
    modelsJobEl.innerHTML = "";
    modelsJobEl.hidden = !job;
    if (!job) return;

    const header = document.createElement("div");
    header.className = "models-job-header";
    const title = document.createElement("span");
    title.textContent = job.spec;
    header.appendChild(title);

    const p = job.progress || {};
    if (job.state === "running") {
      const summary = document.createElement("span");
      summary.className = "models-subtle";
      const eta = formatEta(p.eta_secs);
      summary.textContent =
        `${p.percent ?? 0}% — ${formatBytes(p.done_bytes || 0)} of ` +
        `${formatBytes(p.total_bytes || 0)}${eta ? `, ETA ${eta}` : ""}`;
      header.appendChild(summary);
    } else {
      const state = document.createElement("span");
      state.className = job.state === "failed" ? "models-job-failed" : "models-subtle";
      state.textContent = job.message || job.state;
      header.appendChild(state);
      // Only a finished job can be dismissed — there is no cancellation, and
      // a dismiss button that hid a live download would be a lie.
      header.appendChild(
        iconButton(ICON.close, "Dismiss", () =>
          modelsAction("Dismissing", () => fetch("/api/models/job", { method: "DELETE" })),
        ),
      );
    }
    modelsJobEl.appendChild(header);

    if (job.state === "running") {
      modelsJobEl.appendChild(progressBar(p.percent ?? 0));
      for (const file of p.files || []) {
        const line = document.createElement("div");
        line.className = "models-job-file";
        const name = document.createElement("span");
        name.className = "models-job-file-name";
        name.textContent = file.label;
        name.title = file.label;
        const status = document.createElement("span");
        status.className = "models-subtle";
        const percent = file.size ? Math.min(100, Math.floor((file.downloaded * 100) / file.size)) : 0;
        status.textContent =
          file.state === "done"
            ? "100%"
            : file.state === "retrying"
              ? `${percent}% (retry ${file.retry})`
              : file.state === "queued"
                ? "queued"
                : `${percent}%`;
        line.append(name, status);
        modelsJobEl.appendChild(line);
      }
    }
  }

  function progressBar(percent) {
    const track = document.createElement("div");
    track.className = "models-progress";
    const fill = document.createElement("div");
    fill.className = "models-progress-fill";
    fill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    track.appendChild(fill);
    return track;
  }

  // `orangu-server list`, as a table: the same five columns, the same
  // strings (the server sends `quant`/`size`/`supported` already formatted
  // by the CLI's own code), the same greying of a row this build can't
  // load, the same `(Refresh)` marker on a row whose repo has moved on, and
  // the same `error:` row replacing the cells for a file whose header
  // wouldn't parse. Two icons per row: Show and Delete.
  function renderTable(models, canLoad, canDelete, loading) {
    modelsTableEl.innerHTML = "";
    if (models.length === 0) {
      const empty = document.createElement("div");
      empty.className = "models-empty";
      empty.textContent = "No models here yet — download one above.";
      modelsTableEl.appendChild(empty);
      return;
    }

    const table = document.createElement("table");
    const head = document.createElement("thead");
    head.innerHTML =
      "<tr><th class='num'>NR</th><th>MODEL</th><th>QUANT</th>" +
      "<th class='num'>SIZE</th><th>SUPPORTED</th><th></th></tr>";
    table.appendChild(head);
    const body = document.createElement("tbody");

    for (const model of models) {
      const row = document.createElement("tr");
      if (model.loaded) row.classList.add("loaded");
      // `list` greys a row it can't load; an error row is greyed too, since
      // there is nothing there to load either.
      if (!model.loadable || model.error) row.classList.add("unsupported");

      const nr = document.createElement("td");
      nr.className = "num";
      nr.textContent = model.nr;

      const label = document.createElement("td");
      label.className = "models-label";
      label.textContent = model.label;
      label.title = model.path;
      if (model.loaded) {
        const badge = document.createElement("span");
        badge.className = "models-badge loaded";
        badge.textContent = "loaded";
        badge.title = "The model this server is serving";
        label.appendChild(badge);
      }
      if (model.refresh) {
        const badge = document.createElement("span");
        badge.className = "models-badge";
        badge.textContent = "Refresh";
        badge.title =
          "This repo has a newer revision on Hugging Face — " +
          "`orangu-server refresh` downloads it again";
        label.appendChild(badge);
      }

      row.append(nr, label);

      if (model.error) {
        // `list` drops SIZE and SUPPORTED entirely for an unreadable file
        // and prints the error across the rest of the line.
        const error = document.createElement("td");
        error.colSpan = 3;
        error.textContent = `error: ${model.error}`;
        error.title = model.error;
        error.className = "models-row-error";
        row.appendChild(error);
      } else {
        const quant = document.createElement("td");
        quant.textContent = model.quant;

        const size = document.createElement("td");
        size.className = "num";
        size.textContent = model.size;

        const supported = document.createElement("td");
        supported.textContent = model.supported;

        row.append(quant, size, supported);
      }

      const actions = document.createElement("td");
      actions.className = "models-actions";

      // Nothing at all when `[web].reexec` is off (or the platform has no
      // execve): no Load button, and no check mark in its place either —
      // which row is serving is already on the row itself, as the `loaded`
      // badge beside its name. A column of permanently dead buttons explains
      // nothing that the config file doesn't explain better.
      if (canLoad) {
        if (model.loaded) {
          const marker = iconButton(ICON.loaded, "Currently loaded", null, "is-loaded");
          marker.disabled = true;
          actions.appendChild(marker);
        } else {
          const load = iconButton(ICON.load, `Load ${rowTitle(model)}`, () => {
            if (
              !window.confirm(
                `Load "${rowTitle(model)}"?\n\nThe server restarts itself on this model. ` +
                  `Chat history is kept; anything still generating is not.`,
              )
            ) {
              return;
            }
            loadModel(model);
          });
          // Disabled, not removed, for the two reasons that *are* conditional
          // — a handover already running, and a model this build can't load —
          // so the button says which rather than the click discovering it.
          if (loading) {
            load.disabled = true;
            load.title = `Already loading ${loading}`;
          } else if (!model.loadable) {
            load.disabled = true;
            load.title = "This build cannot load this model";
          }
          load.setAttribute("aria-label", load.title);
          actions.appendChild(load);
        }
      }

      actions.appendChild(
        iconButton(ICON.show, `Show ${model.label}`, () =>
          showMetadata(String(model.nr), rowTitle(model)),
        ),
      );

      // Same for Delete: switched off in the config means no button, not a
      // dead one.
      if (!canDelete) {
        row.appendChild(actions);
        body.appendChild(row);
        continue;
      }

      const remove = iconButton(
        ICON.trash,
        `Delete ${model.label}`,
        () => {
          if (
            !window.confirm(
              `Delete "${rowTitle(model)}" (${model.size})?\n\nThis cannot be undone.`,
            )
          ) {
            return;
          }
          modelsAction(`Deleting ${model.label}`, () =>
            fetch("/api/models", {
              method: "DELETE",
              headers: { "Content-Type": "application/json" },
              // Addressed by `nr`, not by label: a repo with several
              // quantizations on disk prints the same bare MODEL on every
              // one of their rows, so a label would delete whichever came
              // first rather than the row that was clicked. `path` is what
              // the server checks the row against — see ModelRequest::path
              // in web/models.rs.
              body: JSON.stringify({ model: String(model.nr), path: model.path }),
            }),
          );
        },
        "danger",
      );
      // The loaded model's weights are mapped by the running engine, so
      // deleting the file would leave it reading something with no name.
      remove.disabled = model.loaded;
      if (model.loaded) {
        remove.title = "This is the model this server is serving";
        remove.setAttribute("aria-label", remove.title);
      }
      actions.appendChild(remove);

      row.appendChild(actions);
      body.appendChild(row);
    }
    table.appendChild(body);
    modelsTableEl.appendChild(table);
  }

  // How a row is named in a confirmation dialog and above its metadata:
  // MODEL:QUANT, since MODEL alone can name several rows.
  function rowTitle(model) {
    return model.quant && model.quant !== "-" ? `${model.label}:${model.quant}` : model.label;
  }

  // Load: the server replaces itself with a new process serving the chosen
  // model (see reexec.rs). Its listening socket survives the exec, so the
  // port never goes away — but the connection this page has open at that
  // moment does not, and neither does the response to the request that
  // asked for it. So both "got the 202" and "the connection died" mean the
  // same thing here, and both are followed by waiting for the new image.
  async function loadModel(model) {
    if (modelsState.busy) return;
    modelsState.busy = true;
    modelsState.handingOver = true;
    modelsOverlay.classList.add("busy");
    modelsNotice(`Loading ${rowTitle(model)}…`);
    const before = model.path;
    try {
      const res = await fetch("/api/models/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // By `nr`, like every other row action — see the Delete button.
        body: JSON.stringify({ model: String(model.nr), path: model.path }),
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        modelsNotice(detail.trim() || `failed (${res.status})`, "error");
        return;
      }
    } catch {
      // Reset before the reply arrived — the handover beat it. Fall through
      // and find out from the new image.
    }
    const current = await waitForServer();
    if (!current) {
      modelsNotice(
        "The server did not come back. It may still be loading, or the model may have failed " +
          "to load and it fell back — check its output.",
        "error",
      );
    } else if (current.path === before) {
      modelsNotice(`Now serving ${current.display}.`);
    } else {
      // Came back on something else: the load failed and the fallback took
      // over (see ORANGU_FALLBACK_MODEL in reexec.rs).
      modelsNotice(
        `${rowTitle(model)} did not load — the server fell back to ${current.display}.`,
        "error",
      );
    }
    modelsState.busy = false;
    modelsState.handingOver = false;
    modelsOverlay.classList.remove("busy");
    await refreshModels(true).catch(() => {});
  }

  // Polls until the replacement process answers, and returns what it is
  // serving. The wait is generous because a cold load of a large model on a
  // slow disk genuinely takes minutes — and it costs nothing to wait, since
  // the alternative is telling the user it failed while it is still working.
  const HANDOVER_TIMEOUT_MS = 300000;

  async function waitForServer() {
    const deadline = Date.now() + HANDOVER_TIMEOUT_MS;
    // The old image is still answering for the grace period before it execs,
    // so don't accept the very first reply as proof it came back.
    await new Promise((resolve) => setTimeout(resolve, 1000));
    while (Date.now() < deadline) {
      try {
        const res = await fetch("/api/models", { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          if (!data.loading) return data.current;
        }
      } catch {
        // Not accepting yet — the exec is in flight, or the model is loading.
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    return null;
  }

  async function refreshModels(rescan) {
    const res = await fetch(`/api/models${rescan ? "?rescan=true" : ""}`, { cache: "no-store" });
    if (!res.ok) {
      modelsNotice(`could not read the models directory (${res.status})`, "error");
      return;
    }
    const data = await res.json();
    const dir = [
      data.models_dir,
      data.disk_used_bytes != null ? `${formatBytes(data.disk_used_bytes)} used` : null,
      data.disk_available_bytes != null ? `${formatBytes(data.disk_available_bytes)} free` : null,
    ]
      .filter(Boolean)
      .join(" · ");
    modelsDirEl.textContent = dir;
    modelsDirEl.title = data.models_dir;
    // The Hub lookup runs on its own schedule (see refreshUpdateBadges), so
    // its answer is merged onto whatever listing is current.
    for (const model of data.models) {
      model.refresh = modelsState.behind.has(model.nr);
    }
    renderCurrent(data.current, data.loading);
    renderJob(data.job);
    // `can_load`/`loading` change what a row's Load button does, so they
    // are part of what the table is compared on, not just the rows.
    const signature = JSON.stringify([
      data.models,
      data.can_load,
      data.can_delete,
      data.loading ?? null,
    ]);
    if (signature !== modelsState.tableSignature) {
      modelsState.tableSignature = signature;
      renderTable(data.models, data.can_load, data.can_delete, data.loading);
    }
    // The topbar name follows the loaded model, so a handover is visible
    // without reloading the page.
    document.getElementById("model-name").textContent = data.current.display;
  }

  // `list`'s own `(Refresh)` marker. Once per panel opening rather than on
  // every poll: it is one network round trip per distinct repo, and a repo
  // does not gain a new revision while someone is looking at a table. A
  // failure is silent — "unknown" is not "behind", the same rule
  // `orangu-server list` follows.
  async function refreshUpdateBadges() {
    try {
      const res = await fetch("/api/models/updates", { cache: "no-store" });
      if (!res.ok) return;
      const { behind } = await res.json();
      modelsState.behind = new Set(behind || []);
      if (modelsState.open) await refreshModels(false);
    } catch {
      // Offline, or the Hub is unreachable — leave every badge off.
    }
  }

  // `model` is the row's NR (exact — see the load button above); `title` is
  // what to call it on screen.
  async function showMetadata(model, title) {
    modelsState.metadata.model = model;
    modelsState.metadata.title = title ?? modelsState.metadata.title;
    modelsMetadataEl.hidden = false;
    modelsMetadataTitle.textContent = modelsState.metadata.title;
    modelsMetadataBody.textContent = "Loading…";
    const params = new URLSearchParams({ model });
    if (modelsState.metadata.tensors) params.set("tensors", "true");
    if (modelsState.metadata.full) params.set("full", "true");
    try {
      const res = await fetch(`/api/models/metadata?${params}`, { cache: "no-store" });
      const text = await res.text();
      modelsState.metadata.text = res.ok ? text : "";
      modelsMetadataBody.textContent = res.ok ? text : `Could not read it: ${text.trim()}`;
      modelsMetadataBody.scrollTop = 0;
    } catch (err) {
      modelsMetadataBody.textContent = String(err);
    }
  }

  function toggleMetadataOption(key, button) {
    modelsState.metadata[key] = !modelsState.metadata[key];
    button.setAttribute("aria-pressed", modelsState.metadata[key] ? "true" : "false");
    button.classList.toggle("pressed", modelsState.metadata[key]);
    if (modelsState.metadata.model) {
      showMetadata(modelsState.metadata.model).catch((err) => console.error(err));
    }
  }

  modelsMetadataTensorsBtn.addEventListener("click", () =>
    toggleMetadataOption("tensors", modelsMetadataTensorsBtn),
  );
  modelsMetadataFullBtn.addEventListener("click", () =>
    toggleMetadataOption("full", modelsMetadataFullBtn),
  );
  modelsMetadataSaveBtn.addEventListener("click", () => {
    if (!modelsState.metadata.text) return;
    const name = (modelsState.metadata.title || "model").replace(/[^\w.-]+/g, "-");
    downloadTextFile(modelsState.metadata.text, `orangu-${name}-metadata`);
  });
  modelsMetadataCloseBtn.addEventListener("click", () => {
    modelsMetadataEl.hidden = true;
    modelsState.metadata.model = null;
  });

  modelsDownloadForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const repo = modelsDownloadInput.value.trim();
    if (!repo) return;
    modelsDownloadInput.value = "";
    modelsAction(`Starting download of ${repo}`, () =>
      fetch("/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo }),
      }),
    ).catch((err) => console.error(err));
  });

  function openModels() {
    modelsState.open = true;
    modelsOverlay.hidden = false;
    modelsBtn.setAttribute("aria-expanded", "true");
    modelsNotice("");
    refreshModels(true).catch((err) => console.error(err));
    refreshUpdateBadges().catch((err) => console.error(err));
    modelsState.timer = setInterval(() => {
      // `loadModel` drives its own polling across a handover, and every
      // request in that window is expected to fail.
      if (modelsState.handingOver) return;
      refreshModels(false).catch((err) => console.error(err));
    }, MODELS_POLL_MS);
  }

  function closeModels() {
    modelsState.open = false;
    modelsOverlay.hidden = true;
    modelsBtn.setAttribute("aria-expanded", "false");
    modelsMetadataEl.hidden = true;
    modelsState.metadata.model = null;
    if (modelsState.timer) {
      clearInterval(modelsState.timer);
      modelsState.timer = null;
    }
  }

  modelsBtn.addEventListener("click", () => {
    if (modelsOverlay.hidden) {
      openModels();
    } else {
      closeModels();
    }
  });
  modelsCloseBtn.addEventListener("click", closeModels);
  modelsReloadBtn.addEventListener("click", () => {
    modelsNotice("");
    refreshModels(true).catch((err) => console.error(err));
    refreshUpdateBadges().catch((err) => console.error(err));
  });
  // Clicking the backdrop closes; clicking inside the panel does not. Escape
  // closes the metadata viewer first, if it's open, so one key doesn't throw
  // away two levels at once.
  modelsOverlay.addEventListener("click", (event) => {
    if (event.target === modelsOverlay) closeModels();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || modelsOverlay.hidden) return;
    if (!modelsMetadataEl.hidden) {
      modelsMetadataEl.hidden = true;
      modelsState.metadata.model = null;
    } else {
      closeModels();
    }
  });

  document.addEventListener("click", (event) => {
    if (
      !historyPanel.hidden &&
      !historyPanel.contains(event.target) &&
      // historyBtn.contains(), not `!== historyBtn` — a click lands on the
      // button's inner <svg>/<path>, never the <button> element itself, so
      // the strict equality check always treated it as an outside click
      // and closed the panel the instant showHistory() had just opened it.
      !historyBtn.contains(event.target)
    ) {
      hideHistory();
    }
  });

  // MCP inventory is deliberately read-only. The server reads its MCP
  // sections at startup, so changing one belongs in the config plus restart.
  const mcpsBtn = document.getElementById("mcps-btn");
  const mcpsOverlay = document.getElementById("mcps-overlay");
  const mcpsCloseBtn = document.getElementById("mcps-close-btn");
  const mcpsTable = document.getElementById("mcps-table");
  const mcpsDetails = document.getElementById("mcps-details");

  function closeMcps() {
    mcpsOverlay.hidden = true;
    mcpsBtn.setAttribute("aria-expanded", "false");
  }

  async function showMcp(name) {
    const res = await fetch(`/api/mcps/${encodeURIComponent(name)}`, { cache: "no-store" });
    if (!res.ok) throw new Error(await res.text());
    mcpsDetails.textContent = JSON.stringify(await res.json(), null, 2);
    mcpsDetails.hidden = false;
  }

  async function openMcps() {
    mcpsOverlay.hidden = false;
    mcpsBtn.setAttribute("aria-expanded", "true");
    mcpsDetails.hidden = true;
    mcpsTable.textContent = "Loading…";
    const res = await fetch("/api/mcps", { cache: "no-store" });
    if (!res.ok) throw new Error(await res.text());
    const mcps = await res.json();
    if (!mcps.length) {
      mcpsTable.textContent = "No MCP servers are configured.";
      return;
    }
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Name</th><th>Endpoint</th><th>Status</th><th></th></tr></thead>";
    const body = document.createElement("tbody");
    mcps.forEach((mcp) => {
      const row = document.createElement("tr");
      [mcp.name, mcp.endpoint, mcp.enabled ? "Enabled" : "Disabled"].forEach((value) => {
        const cell = document.createElement("td"); cell.textContent = value; row.appendChild(cell);
      });
      const action = document.createElement("td");
      const show = document.createElement("button");
      show.type = "button"; show.className = "icon-btn subtle-btn"; show.textContent = "Show";
      show.addEventListener("click", () => showMcp(mcp.name).catch((err) => { mcpsDetails.textContent = err.message; mcpsDetails.hidden = false; }));
      action.appendChild(show); row.appendChild(action); body.appendChild(row);
    });
    table.appendChild(body);
    mcpsTable.replaceChildren(table);
  }

  mcpsBtn.addEventListener("click", () => openMcps().catch((err) => { mcpsTable.textContent = err.message; }));
  mcpsCloseBtn.addEventListener("click", closeMcps);
  mcpsOverlay.addEventListener("click", (event) => { if (event.target === mcpsOverlay) closeMcps(); });

  (async function init() {
    const savedId = localStorage.getItem("orangu-session-id");
    if (savedId) {
      try {
        await loadSession(savedId);
        return;
      } catch {
        // Stale/missing session — fall through to creating a new one.
      }
    }
    await newChat();
  })().catch((err) => console.error(err));
})();
