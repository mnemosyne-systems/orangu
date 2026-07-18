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
  const themeToggleBtn = document.getElementById("theme-toggle-btn");

  const state = { sessionId: null, busy: false, abortController: null };

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

  function addMessage(role, text) {
    const el = document.createElement("div");
    el.className = `message ${role}`;
    el.textContent = text;
    transcript.appendChild(el);
    transcript.scrollTop = transcript.scrollHeight;
    return el;
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

  // Appended once generation finishes (streamed replies only know their
  // own time and raw text at the "done" event; history reloads know both
  // right away from the loaded session) — deliberately its own element
  // rather than baked into the rendered markdown, so it survives
  // `assistantEl.innerHTML = payload.html` reassignments during streaming
  // and never gets treated as message content (copy/paste, markdown
  // re-render, ...).
  function addTimingFooter(assistantEl, ms, rawContent) {
    if (ms == null) return;
    const footer = document.createElement("div");
    footer.className = "gen-time";

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

  // sendBtn stays enabled throughout a request — while idle it submits the
  // form, while busy its click handler (below) cancels the in-flight
  // request instead, so it can't be disabled the way `input` is.
  function setBusy(busy) {
    state.busy = busy;
    input.disabled = busy;
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

  async function newChat() {
    const session = await createSession();
    state.sessionId = session.id;
    localStorage.setItem("orangu-session-id", session.id);
    transcript.innerHTML = "";
    hideHistory();
  }

  async function loadSession(id) {
    const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error(`failed to load session (${res.status})`);
    const session = await res.json();
    state.sessionId = session.id;
    localStorage.setItem("orangu-session-id", session.id);
    transcript.innerHTML = "";
    for (const message of session.messages) {
      if (message.role === "assistant") {
        const el = addRenderedMessage("assistant", message.html || escapeHtml(message.content));
        addTimingFooter(el, message.generation_ms, message.content);
      } else {
        addMessage(message.role, message.content);
      }
    }
    hideHistory();
  }

  function formatDate(unixSeconds) {
    return new Date(unixSeconds * 1000).toLocaleString();
  }

  async function refreshHistory() {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const sessions = await res.json();
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
      const title = document.createElement("div");
      title.className = "history-title";
      title.textContent = session.title || "New chat";
      const date = document.createElement("div");
      date.className = "history-date";
      date.textContent = formatDate(session.updated_at);
      item.appendChild(title);
      item.appendChild(date);
      item.addEventListener("click", () => {
        loadSession(session.id).catch((err) => console.error(err));
      });
      historyList.appendChild(item);
    }
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
  const FAILURE_MESSAGE = "🦧⚙️";

  function showFailure(assistantEl, consoleLabel, detail) {
    console.error(consoleLabel, detail);
    assistantEl.className = "message error";
    assistantEl.textContent = FAILURE_MESSAGE;
  }

  async function sendMessage(text) {
    if (!state.sessionId) {
      await newChat();
    }
    addMessage("user", text);
    const assistantEl = addMessage("assistant", "🤖");
    assistantEl.classList.add("pending");
    setBusy(true);
    const controller = new AbortController();
    state.abortController = controller;

    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: text }),
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
          assistantEl.classList.remove("pending");
          if (payload.type === "token" || payload.type === "done") {
            assistantEl.innerHTML = payload.html;
            pinCodeBlocksToLatest(assistantEl);
            if (payload.type === "done") {
              if (payload.truncated) {
                const notice = document.createElement("p");
                notice.className = "truncated-notice";
                notice.textContent = "⚠️ Response was cut off at the token limit.";
                assistantEl.appendChild(notice);
              }
              addTimingFooter(assistantEl, payload.generation_ms, payload.content);
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
    if (!text) return;
    input.value = "";
    sendMessage(text).catch((err) => console.error(err));
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
