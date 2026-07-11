(() => {
  "use strict";

  const transcript = document.getElementById("transcript");
  const input = document.getElementById("input");
  const composer = document.getElementById("composer");
  const sendBtn = document.getElementById("send-btn");
  const newChatBtn = document.getElementById("new-chat-btn");
  const historyBtn = document.getElementById("history-btn");
  const historyPanel = document.getElementById("history-panel");
  const historyList = document.getElementById("history-list");
  const themeToggleBtn = document.getElementById("theme-toggle-btn");

  const state = { sessionId: null, busy: false };

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

  function setBusy(busy) {
    state.busy = busy;
    input.disabled = busy;
    sendBtn.disabled = busy;
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
        addRenderedMessage("assistant", message.html || escapeHtml(message.content));
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
    const assistantEl = addMessage("assistant", "");
    setBusy(true);

    let buffer = "";
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: text }),
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
          if (payload.type === "token") {
            buffer += payload.text;
            assistantEl.textContent = buffer;
            transcript.scrollTop = transcript.scrollHeight;
          } else if (payload.type === "done") {
            assistantEl.innerHTML = payload.html;
            transcript.scrollTop = transcript.scrollHeight;
          } else if (payload.type === "error") {
            showFailure(assistantEl, "orangu-server generation error:", payload.message);
          }
        }
      }
    } catch (err) {
      showFailure(assistantEl, "orangu-server request failed:", err);
    } finally {
      setBusy(false);
    }
  }

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
      event.target !== historyBtn
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
