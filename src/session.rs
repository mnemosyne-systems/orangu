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

use crate::{
    config::LlmConfiguration,
    llm::{ChatMessage, LlmResponse, OpenAiClient, SlotRegistry, StreamMetrics},
    tools::ToolExecutor,
};
use anyhow::{Result, anyhow};

/// The contextual fragment compaction expires: a snapshot of the working tree
/// that was worth its tokens on the turn it announced and worth nothing after.
const WORLD_STATE_TAG: &str = "world_state_changes";

/// The opening-tag prefix, used as a cheap "is there anything to strip here"
/// test before doing any real work.
const FRAGMENT_OPEN: &str = "<world_state_changes";

/// What an evicted tool output is replaced with. Compared against as well as
/// written, so a second compaction pass does not count an already-evicted
/// message as more bytes it could reclaim.
const EVICTED_TOOL_OUTPUT: &str = "[Tool output evicted to save tokens]";

/// Replace a `<tag>…</tag>` contextual fragment inside `content` with a short
/// stub, in place. Matches [`crate::context::fragments::ContextualFragment`]'s
/// rendering, including the attribute form (`<tag k="v">`).
///
/// Only whole, well-formed fragments are touched; anything else is left
/// exactly as it was, so a user message that merely mentions the tag name is
/// never mangled.
fn strip_stale_fragment(content: &mut String, tag: &str) {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(content.len());
    let mut rest = content.as_str();
    let mut stripped = false;
    while let Some(start) = rest.find(&open) {
        // The opening tag must actually end (`>`) before the closing tag does.
        let after = &rest[start..];
        let Some(head_end) = after.find('>') else {
            break;
        };
        let Some(end) = after.find(&close) else { break };
        if end < head_end {
            break;
        }
        out.push_str(&rest[..start]);
        out.push_str(&format!("[{tag} from an earlier turn, dropped]"));
        rest = &after[end + close.len()..];
        stripped = true;
    }
    if stripped {
        out.push_str(rest);
        *content = out;
    }
}

pub struct ChatSession {
    messages: Vec<ChatMessage>,
    /// Cached LLM client, reused across prompts so the underlying HTTP
    /// connection pool survives between requests. Rebuilt only when the
    /// profile fields that shape the client change.
    client: Option<(ClientKey, OpenAiClient)>,

    /// The `id_slot` registry to pin this session's requests through, set via
    /// [`Self::with_slots`]. `None` (the default) means this session never
    /// pins a slot — the right choice for scratch/one-shot sessions
    /// (`/auto_review`, `explorer.rs`, tests) that gain nothing from it and
    /// would otherwise each pay a redundant `/props` probe.
    slots: Option<SlotRegistry>,
    /// This session's currently assigned slot, and the endpoint it was
    /// assigned for — re-resolved by [`Self::ensure_slot_assigned`] whenever
    /// the profile's endpoint changes (e.g. after `/server`).
    assigned_slot: Option<u32>,
    assigned_slot_endpoint: Option<String>,
    /// A plain client used only for the (at most once per endpoint) `/props`
    /// probe behind slot assignment — distinct from `client`'s
    /// [`OpenAiClient`], which is rebuilt on profile changes and not exposed.
    probe_client: reqwest::Client,

    pub model_verbosity_override: Option<String>,
}

/// The subset of [`LlmConfiguration`] that the [`OpenAiClient`] is built from.
/// Two profiles producing the same key yield an interchangeable client.
#[derive(PartialEq, Eq)]
struct ClientKey {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    request_timeout_seconds: u64,
    code_max_tokens: u32,
}

impl ClientKey {
    fn from_profile(profile: &LlmConfiguration) -> Self {
        Self {
            endpoint: profile.endpoint.clone(),
            model: profile.model.clone(),
            api_key: profile.api_key.clone(),
            request_timeout_seconds: profile.request_timeout_seconds,
            code_max_tokens: profile.code_max_tokens,
        }
    }
}

impl ChatSession {
    pub fn new(system_prompt: &str) -> Self {
        Self {
            messages: vec![ChatMessage::system(system_prompt)],
            client: None,
            slots: None,
            assigned_slot: None,
            assigned_slot_endpoint: None,
            probe_client: reqwest::Client::new(),

            model_verbosity_override: None,
        }
    }

    /// Attach a shared [`SlotRegistry`] so this session's requests pin to a
    /// specific orangu-server `id_slot`. Only the interactive per-tab session
    /// should call this — see the `slots` field doc.
    pub fn with_slots(mut self, slots: SlotRegistry) -> Self {
        self.slots = Some(slots);
        self
    }

    pub fn assigned_slot(&self) -> Option<u32> {
        self.assigned_slot
    }

    /// (Re)resolve `assigned_slot` for `profile`'s endpoint if a
    /// [`SlotRegistry`] is attached and the endpoint changed since the last
    /// assignment (e.g. a `/server` switch). A no-op returning `None` when no
    /// registry is attached. Called automatically by [`Self::prompt`] /
    /// [`Self::prompt_without_tools`]; Feature C's tab-activate/resume path
    /// calls it explicitly to resolve a slot before attempting a restore.
    pub async fn ensure_slot_assigned(
        &mut self,
        profile: &LlmConfiguration,
        client: &reqwest::Client,
    ) -> Option<u32> {
        let slots = self.slots.as_ref()?;
        if self.assigned_slot_endpoint.as_deref() != Some(profile.endpoint.as_str()) {
            self.assigned_slot = slots
                .assign_slot(client, &profile.endpoint, profile.api_key.as_deref())
                .await;
            self.assigned_slot_endpoint = Some(profile.endpoint.clone());
        }
        self.assigned_slot
    }

    /// Replace this session's system prompt, in the system message where it
    /// belongs.
    ///
    /// Mid-conversation this used to *append* the new prompt as a **user**
    /// message prefixed `[System Update]`. That kept the server's prefix cache
    /// intact, which is the one thing it had going for it, and was wrong in
    /// every other way: the model was handed its own instructions in the
    /// user's voice, the original system message still carried the superseded
    /// prompt so the two contradicted each other, and a second `/verbosity`
    /// appended a third copy.
    ///
    /// Rewriting message zero does cost a re-prefill of the whole
    /// conversation — see [`Self::compact_transcript`] for why that is not
    /// done lightly. It is affordable here because of *when* this runs: only
    /// from `/server` and `/verbosity`,
    /// both explicit one-off commands. `/server` changes the endpoint, so that
    /// server's cache is cold regardless and the rewrite is free; `/verbosity`
    /// pays once, for a command whose entire purpose is to change how the
    /// model behaves from here on.
    ///
    /// An unchanged prompt is left strictly alone, so re-selecting the server
    /// you are already on — which `/server` explicitly supports — costs
    /// nothing.
    ///
    /// Appending a real `system` message instead was the other candidate and
    /// is not portable: several widely-used templates (Mistral's among them)
    /// call `raise_exception` on any system message that is not the first,
    /// which would turn a `/verbosity` into a failed request.
    pub fn set_system_prompt(&mut self, prompt: &str) {
        match self.messages.first_mut() {
            Some(message) if message.role == "system" => {
                if message.content != prompt {
                    message.content = prompt.to_string();
                }
            }
            _ => self.messages.insert(0, ChatMessage::system(prompt)),
        }
    }

    pub fn clear(&mut self, system_prompt: &str) {
        self.messages.clear();
        self.messages.push(ChatMessage::system(system_prompt));
    }

    pub fn push_user(&mut self, content: &str) {
        self.messages.push(ChatMessage::user(content));
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn restore(&mut self, messages: Vec<ChatMessage>) {
        self.messages = messages;
    }

    pub fn checkpoint(&self) -> usize {
        self.messages.len()
    }

    pub fn rollback(&mut self, checkpoint: usize) {
        self.messages.truncate(checkpoint);
    }

    /// Floor on a compaction pass: below this there is nothing worth the
    /// disruption, however the ratio below works out.
    const COMPACT_MIN_RECLAIM_BYTES: usize = 4 * 1024;

    /// Compaction runs only when it can reclaim at least this fraction of the
    /// transcript — `2` meaning half of it.
    ///
    /// Compaction is not free on the server side. It rewrites history the
    /// server has already prefilled, and the prefix cache matches on a token-id
    /// *prefix*, so the first rewritten message forces everything after it to
    /// be processed again. Worse, evicting a message **shrinks** it, which drags
    /// the next turn's divergence point back to nearly the start of the
    /// transcript. Compacting whenever something became eligible — one message
    /// per turn, forever — measured **3.5× more prefill** over fourteen
    /// tool-using turns than never compacting at all, with nine consecutive
    /// turns reusing essentially nothing from cache.
    ///
    /// A fixed byte threshold is the wrong shape for this: too low and it is
    /// the old behaviour, too high and eviction never happens and the context
    /// grows without bound. A *ratio* is self-tuning. Paying one whole
    /// re-prefill to halve the transcript means the next pass cannot come due
    /// until it has roughly doubled again, so evictions get geometrically
    /// rarer as a conversation grows instead of arriving every turn.
    const COMPACT_MIN_RECLAIM_RATIO: usize = 2;

    /// Rewrite the transcript to drop what is no longer worth its tokens —
    /// but only once there is enough to drop to justify the re-prefill it
    /// costs. See [`Self::COMPACT_MIN_RECLAIM_RATIO`].
    pub fn compact_transcript(&mut self) {
        let reclaimable = self.compactable_bytes();
        if reclaimable < Self::COMPACT_MIN_RECLAIM_BYTES {
            return;
        }
        let transcript: usize = self.messages.iter().map(|m| m.content.len()).sum();
        if reclaimable * Self::COMPACT_MIN_RECLAIM_RATIO < transcript {
            return;
        }
        self.apply_compaction();
    }

    /// What [`Self::apply_compaction`] would remove, in bytes, without
    /// touching anything.
    fn compactable_bytes(&self) -> usize {
        let mut total = 0usize;
        let mut user_turns = 0;
        for msg in self.messages.iter().rev() {
            if msg.role == "user" {
                user_turns += 1;
                // Cheap gate first: only a message that actually carries a
                // fragment is worth copying to measure.
                if msg.content.contains(FRAGMENT_OPEN) {
                    let mut copy = msg.content.clone();
                    strip_stale_fragment(&mut copy, WORLD_STATE_TAG);
                    total += msg.content.len().saturating_sub(copy.len());
                }
            } else if Self::is_evictable_tool(msg, user_turns) {
                total += msg.content.len() - EVICTED_TOOL_OUTPUT.len();
            }
        }
        total
    }

    fn is_evictable_tool(msg: &ChatMessage, user_turns: usize) -> bool {
        msg.role == "tool"
            && user_turns > 3
            && msg.content.len() > 500
            && msg.content != EVICTED_TOOL_OUTPUT
    }

    fn apply_compaction(&mut self) {
        let mut user_turns = 0;
        for msg in self.messages.iter_mut().rev() {
            if msg.role == "user" {
                user_turns += 1;
                // A `world_state_changes` fragment describes what the working
                // tree looked like at one moment. It is worth its tokens on
                // the turn it announces and worth nothing afterwards — but
                // being part of a user message, nothing ever removed it, so
                // every turn's snapshot stayed in the transcript for the rest
                // of the session and was re-sent forever. A session that once
                // took an oversized fragment could not recover from it, even
                // after the code that produced it was fixed.
                //
                // Every user message present here is from an earlier turn:
                // compaction runs *before* [`Self::prompt`] appends the one
                // being sent now, so there is no live fragment to protect.
                strip_stale_fragment(&mut msg.content, WORLD_STATE_TAG);
            } else if Self::is_evictable_tool(msg, user_turns) {
                msg.content = EVICTED_TOOL_OUTPUT.to_string();
            }
        }
    }

    /// One-shot prompt with no tool definitions and a capped response length:
    /// a single chat round. The model cannot start tool-call rounds and cannot
    /// generate unbounded output — for requests whose prompt is self-contained
    /// (the content to work on is inline), such as `/auto_review`.
    ///
    /// `on_text_delta` and `on_stream_metrics` are forwarded to the streaming
    /// client, which fires them as the response arrives — they drive the live
    /// status display. The complete text is also returned at the end.
    /// A `max_response_tokens` of `0` disables the cap.
    pub async fn prompt_without_tools<F, G>(
        &mut self,
        user_input: &str,
        profile: &LlmConfiguration,
        max_response_tokens: u32,
        mut on_text_delta: F,
        mut on_stream_metrics: G,
    ) -> Result<String>
    where
        F: FnMut(&str),
        G: FnMut(StreamMetrics),
    {
        let probe_client = self.probe_client.clone();
        let id_slot = self.ensure_slot_assigned(profile, &probe_client).await;
        // Built per call rather than cached: the cached client is keyed for
        // the tool-enabled flow and carries that flow's response cap, not
        // this request's.
        let client = OpenAiClient::from_profile(profile)?
            .with_max_tokens(max_response_tokens)
            .with_id_slot(id_slot);
        let checkpoint = self.checkpoint();
        self.messages.push(ChatMessage::user(user_input));
        match client
            .chat(
                &self.messages,
                &[],
                &mut on_text_delta,
                &mut on_stream_metrics,
            )
            .await
        {
            Ok(LlmResponse::Text(text)) => {
                self.messages.push(ChatMessage::assistant(&text));
                Ok(text)
            }
            Ok(LlmResponse::ToolCalls(_)) => {
                self.rollback(checkpoint);
                Err(anyhow!("the model attempted a tool call without tools"))
            }
            Err(err) => {
                self.rollback(checkpoint);
                Err(err)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn prompt<F, G, H, I, J>(
        &mut self,
        user_input: &str,
        profile: &LlmConfiguration,
        tools: &ToolExecutor,
        mut on_text_delta: F,
        mut on_stream_metrics: G,
        mut on_tool_running: H,
        mut on_tool_call: I,
        mut on_mcp_approval: J,
    ) -> Result<String>
    where
        F: FnMut(&str),
        G: FnMut(StreamMetrics),
        H: FnMut(bool),
        I: FnMut(&crate::llm::ToolCall),
        J: FnMut(&crate::llm::ToolCall) -> bool,
    {
        self.compact_transcript();

        let key = ClientKey::from_profile(profile);
        if self
            .client
            .as_ref()
            .is_none_or(|(cached, _)| *cached != key)
        {
            self.client = Some((key, OpenAiClient::from_profile(profile)?));
        }
        let probe_client = self.probe_client.clone();
        let id_slot = self.ensure_slot_assigned(profile, &probe_client).await;
        // Cheap clone: shares the underlying reqwest connection pool.
        // `id_slot` is applied to the clone, not the cached client, since the
        // assignment can change (e.g. after `/server`) without invalidating
        // the connection pool `ClientKey` intentionally leaves it out of.
        let client = self
            .client
            .as_ref()
            .expect("client populated above")
            .1
            .clone()
            .with_id_slot(id_slot);
        let tool_definitions = tools.definitions();
        let checkpoint = self.checkpoint();
        self.messages.push(ChatMessage::user(user_input));

        for _ in 0..profile.max_tool_rounds {
            match client
                .chat(
                    &self.messages,
                    &tool_definitions,
                    &mut on_text_delta,
                    &mut on_stream_metrics,
                )
                .await
            {
                Ok(response) => match response {
                    LlmResponse::Text(text) => {
                        self.messages.push(ChatMessage::assistant(&text));
                        return Ok(text);
                    }
                    LlmResponse::ToolCalls(tool_calls) => {
                        self.messages
                            .push(ChatMessage::assistant_tool_calls(tool_calls.clone()));

                        on_tool_running(true);
                        for tool_call in tool_calls {
                            on_tool_call(&tool_call);
                            let rendered = if tools.requires_mcp_approval(&tool_call.function.name)
                                && !on_mcp_approval(&tool_call)
                            {
                                "error: MCP tool call denied by user".to_string()
                            } else {
                                match tools
                                    .execute(
                                        &tool_call.function.name,
                                        &tool_call.function.arguments.into_iter().collect(),
                                    )
                                    .await
                                {
                                    Ok(result) => result,
                                    Err(err) => format!("error: {err:#}"),
                                }
                            };

                            self.messages
                                .push(ChatMessage::tool_result(&tool_call.id, &rendered));
                        }
                        on_tool_running(false);
                    }
                },
                Err(err) => {
                    self.rollback(checkpoint);
                    return Err(err);
                }
            }
        }

        self.rollback(checkpoint);
        Err(anyhow!(
            "model exceeded the configured max_tool_rounds ({})",
            profile.max_tool_rounds
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatSession, strip_stale_fragment};
    use crate::config::LlmConfiguration;
    use crate::context::fragments::ContextualFragment;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// The turn a fragment announces is the only turn it is worth anything
    /// on. Left in place it was re-sent for the rest of the session, so a
    /// session that once took an oversized one could never recover — which is
    /// exactly what a persisted, auto-resumed session did.
    ///
    /// The ordering here is the one [`ChatSession::prompt`] uses: compact the
    /// transcript, *then* append this turn. That is what makes "every
    /// fragment still in the history is stale" true.
    #[test]
    fn an_earlier_turns_world_state_fragment_is_dropped_on_the_next_turn() {
        let fragment = ContextualFragment::new("world_state_changes", &"x".repeat(100_000));
        let mut session = ChatSession::new("system");
        session.push_user(&format!("{}\n\nfirst question", fragment.render()));
        session
            .messages
            .push(crate::llm::ChatMessage::assistant("ok"));

        session.compact_transcript();
        session.push_user(&format!("{}\n\nsecond question", fragment.render()));

        let first = &session.messages()[1].content;
        assert!(!first.contains("xxxx"), "earlier fragment survived");
        assert!(first.contains("first question"), "the question was lost");
        assert!(first.contains("[world_state_changes from an earlier turn, dropped]"));

        // This turn's fragment is the live one and must reach the model.
        let newest = &session.messages()[3].content;
        assert!(newest.contains("xxxx"), "the live fragment was dropped");
        assert!(newest.contains("second question"));
    }

    /// The case that was actually on disk: a single resumed turn carrying a
    /// huge fragment, with nothing after it. It must be recoverable.
    #[test]
    fn a_resumed_session_recovers_from_one_oversized_fragment() {
        let fragment = ContextualFragment::new("world_state_changes", &"x".repeat(400_000));
        let mut session = ChatSession::new("system");
        session.push_user(&format!("{}\n\nthe question", fragment.render()));
        session
            .messages
            .push(crate::llm::ChatMessage::assistant(""));

        session.compact_transcript();

        let total: usize = session.messages().iter().map(|m| m.content.len()).sum();
        assert!(total < 1_000, "session still carries {total} bytes");
        assert!(session.messages()[1].content.contains("the question"));
    }

    /// Build a transcript of `turns` tool-using turns, each carrying a tool
    /// output of `tool_bytes`.
    fn tool_conversation(turns: usize, tool_bytes: usize) -> ChatSession {
        let mut session = ChatSession::new("system");
        for t in 0..turns {
            session.push_user(&format!("question {t}"));
            session
                .messages
                .push(crate::llm::ChatMessage::assistant("calling"));
            let mut tool = crate::llm::ChatMessage::assistant("");
            tool.role = "tool".into();
            tool.content = "x".repeat(tool_bytes);
            session.messages.push(tool);
            session
                .messages
                .push(crate::llm::ChatMessage::assistant(&format!("answer {t}")));
        }
        session
    }

    fn evicted_count(session: &ChatSession) -> usize {
        session
            .messages()
            .iter()
            .filter(|m| m.content == super::EVICTED_TOOL_OUTPUT)
            .count()
    }

    /// The regression this batching exists for: rewriting history the server
    /// has already prefilled costs a re-prefill of everything after the edit,
    /// so a backlog too small to be worth that must be left alone.
    #[test]
    fn a_small_backlog_is_not_worth_a_re_prefill_and_is_left_alone() {
        let mut session = tool_conversation(6, 2_000);
        assert!(session.compactable_bytes() < ChatSession::COMPACT_MIN_RECLAIM_BYTES);

        let before: Vec<String> = session
            .messages()
            .iter()
            .map(|m| m.content.clone())
            .collect();
        session.compact_transcript();
        let after: Vec<String> = session
            .messages()
            .iter()
            .map(|m| m.content.clone())
            .collect();

        assert_eq!(before, after, "history was rewritten for a small saving");
        assert_eq!(evicted_count(&session), 0);
    }

    /// And when the backlog *is* worth it — here most of the transcript is old
    /// tool output — the whole of it goes in one pass. One expensive turn, not
    /// one expensive turn per evicted message.
    #[test]
    fn a_backlog_worth_half_the_transcript_is_cleared_in_a_single_pass() {
        let mut session = tool_conversation(12, 8 * 1024);
        let transcript: usize = session.messages().iter().map(|m| m.content.len()).sum();
        assert!(
            session.compactable_bytes() * ChatSession::COMPACT_MIN_RECLAIM_RATIO >= transcript,
            "test transcript does not actually meet the rule under test"
        );

        session.compact_transcript();
        let first = evicted_count(&session);
        assert!(first >= 8, "only {first} evicted in one pass");

        // Nothing new became eligible, so a second pass must be a no-op —
        // otherwise every turn would keep paying.
        let before: Vec<String> = session
            .messages()
            .iter()
            .map(|m| m.content.clone())
            .collect();
        session.compact_transcript();
        let after: Vec<String> = session
            .messages()
            .iter()
            .map(|m| m.content.clone())
            .collect();
        assert_eq!(before, after);
        assert_eq!(evicted_count(&session), first);
    }

    /// An already-evicted stub must not be counted as reclaimable, or the rule
    /// would stay satisfied forever and compaction would run on every single
    /// turn — exactly what it is meant to stop.
    #[test]
    fn an_already_evicted_message_is_not_counted_again() {
        let mut session = tool_conversation(12, 8 * 1024);
        session.compact_transcript();
        assert!(evicted_count(&session) > 0);
        assert!(
            session.compactable_bytes() < ChatSession::COMPACT_MIN_RECLAIM_BYTES,
            "evicted stubs are still being counted as reclaimable"
        );
    }

    /// The shape the ratio exists to produce: after a pass, the transcript has
    /// to grow substantially before another comes due. A rule that fires again
    /// immediately is the old per-turn behaviour wearing a threshold.
    #[test]
    fn a_pass_buys_several_quiet_turns_before_the_next_one() {
        let mut session = tool_conversation(12, 8 * 1024);
        session.compact_transcript();
        let after_first = evicted_count(&session);

        let mut quiet = 0;
        for t in 0..4 {
            session.push_user(&format!("later question {t}"));
            let mut tool = crate::llm::ChatMessage::assistant("");
            tool.role = "tool".into();
            tool.content = "y".repeat(8 * 1024);
            session.messages.push(tool);
            session
                .messages
                .push(crate::llm::ChatMessage::assistant("ok"));
            session.compact_transcript();
            if evicted_count(&session) == after_first {
                quiet += 1;
            }
        }
        assert!(quiet >= 2, "only {quiet} of 4 follow-up turns were free");
    }

    /// The system prompt belongs in the system message, mid-conversation or
    /// not. It used to arrive as a `user` message prefixed `[System Update]`,
    /// leaving the model with two contradicting sets of instructions — one of
    /// them apparently spoken by the user.
    #[test]
    fn a_mid_conversation_system_prompt_replaces_the_system_message() {
        let mut session = ChatSession::new("old instructions");
        session.push_user("a question");
        session
            .messages
            .push(crate::llm::ChatMessage::assistant("an answer"));

        session.set_system_prompt("new instructions");

        let roles: Vec<&str> = session.messages().iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            ["system", "user", "assistant"],
            "a message was added"
        );
        assert_eq!(session.messages()[0].content, "new instructions");
        assert!(
            !session
                .messages()
                .iter()
                .any(|m| m.content.contains("[System Update]")),
            "the instructions leaked into the conversation as a user message"
        );
    }

    /// Re-selecting the server you are already on is something `/server`
    /// explicitly supports, and it must not throw away the conversation's KV
    /// cache for nothing.
    #[test]
    fn an_unchanged_system_prompt_leaves_the_message_untouched() {
        let mut session = ChatSession::new("instructions");
        session.push_user("a question");
        let before = session.messages().to_vec();

        session.set_system_prompt("instructions");

        let after = session.messages();
        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after) {
            assert_eq!(b.role, a.role);
            assert_eq!(b.content, a.content);
        }
    }

    /// A session that somehow has no system message gets one, at the front.
    #[test]
    fn a_session_without_a_system_message_gains_one_at_the_front() {
        let mut session = ChatSession::new("system");
        session.messages.remove(0);
        session.push_user("a question");

        session.set_system_prompt("instructions");

        assert_eq!(session.messages()[0].role, "system");
        assert_eq!(session.messages()[0].content, "instructions");
        assert_eq!(session.messages()[1].role, "user");
    }

    #[test]
    fn a_message_merely_naming_the_tag_is_left_alone() {
        let mut content = "why does <world_state_changes> show up in my prompt?".to_string();
        let before = content.clone();
        strip_stale_fragment(&mut content, "world_state_changes");
        assert_eq!(content, before);
    }

    #[test]
    fn a_fragment_with_attributes_is_still_recognised() {
        let mut content = ContextualFragment::new("world_state_changes", "body")
            .with_attribute("hash", "abc")
            .render();
        strip_stale_fragment(&mut content, "world_state_changes");
        assert_eq!(
            content,
            "[world_state_changes from an earlier turn, dropped]"
        );
    }

    fn test_profile(endpoint: &str) -> LlmConfiguration {
        LlmConfiguration {
            endpoint: endpoint.to_string(),
            model: "test-model".to_string(),
            role: "all".to_string(),
            api_key: None,
            request_timeout_seconds: 5,
            max_tool_rounds: 10,
            review_max_tokens: 512,
            review_confidence_threshold: 80,
            code_max_tokens: 0,
            system_prompt: "".to_string(),
            model_verbosity: None,
        }
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    /// Serve exactly one HTTP request on `listener`, answering with `sse_body`
    /// as a chat-completion event stream, and return the request body that the
    /// client sent.
    fn serve_one_chat_response(
        listener: TcpListener,
        sse_body: &'static str,
    ) -> std::thread::JoinHandle<String> {
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept connection");
            let mut request = Vec::new();
            let mut buffer = [0u8; 4096];
            let header_end = loop {
                let read = stream.read(&mut buffer).expect("read request");
                request.extend_from_slice(&buffer[..read]);
                if let Some(position) = find_subsequence(&request, b"\r\n\r\n") {
                    break position + 4;
                }
            };
            let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
            let content_length: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse().ok())
                .unwrap_or(0);
            while request.len() < header_end + content_length {
                let read = stream.read(&mut buffer).expect("read body");
                request.extend_from_slice(&buffer[..read]);
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                sse_body.len(),
                sse_body,
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
            String::from_utf8_lossy(&request[header_end..]).to_string()
        })
    }

    #[tokio::test]
    async fn prompt_without_tools_returns_text_and_caps_the_response() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let server = serve_one_chat_response(
            listener,
            "data: {\"choices\":[{\"delta\":{\"content\":\"VERDICT: APPROVE\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        );

        let mut session = ChatSession::new("system");
        let mut deltas = Vec::new();
        let response = session
            .prompt_without_tools(
                "review this",
                &test_profile(&endpoint),
                512,
                |delta| deltas.push(delta.to_string()),
                |_| {},
            )
            .await
            .expect("text response");
        assert_eq!(response, "VERDICT: APPROVE");
        // The streamed deltas reach the caller's callback as they arrive.
        assert_eq!(deltas.concat(), "VERDICT: APPROVE");

        // The exchange is recorded like a normal prompt.
        let roles: Vec<&str> = session
            .messages()
            .iter()
            .map(|message| message.role.as_str())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant"]);

        // The request carries the response cap and no tool definitions.
        let body = server.join().expect("server thread");
        assert!(body.contains("\"max_tokens\":512"), "request body: {body}");
        assert!(!body.contains("\"tools\""), "request body: {body}");
    }

    #[tokio::test]
    async fn prompt_without_tools_with_zero_cap_omits_max_tokens() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let server = serve_one_chat_response(
            listener,
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        );

        let mut session = ChatSession::new("system");
        session
            .prompt_without_tools("review this", &test_profile(&endpoint), 0, |_| {}, |_| {})
            .await
            .expect("text response");

        // A zero cap means no cap: the request carries no max_tokens at all.
        let body = server.join().expect("server thread");
        assert!(!body.contains("max_tokens"), "request body: {body}");
    }

    #[tokio::test]
    async fn prompt_without_tools_rolls_back_when_the_model_calls_a_tool() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let server = serve_one_chat_response(
            listener,
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call0\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        );

        let mut session = ChatSession::new("system");
        let error = session
            .prompt_without_tools("review this", &test_profile(&endpoint), 512, |_| {}, |_| {})
            .await
            .expect_err("tool calls are rejected");
        assert!(
            error.to_string().contains("tool call"),
            "unexpected error: {error:#}"
        );
        // The failed exchange is rolled back; only the system prompt remains.
        assert_eq!(session.messages().len(), 1);
        let _ = server.join();
    }

    #[tokio::test]
    async fn prompt_without_tools_rolls_back_on_request_errors() {
        // The server accepts the connection and closes it without sending a
        // response, which fails the request deterministically on every
        // platform (no reliance on a freed port staying unbound).
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept connection");
            drop(stream);
        });

        let mut session = ChatSession::new("system");
        let result = session
            .prompt_without_tools("review this", &test_profile(&endpoint), 512, |_| {}, |_| {})
            .await;
        assert!(result.is_err());
        // The failed exchange is rolled back; only the system prompt remains.
        assert_eq!(session.messages().len(), 1);
        let _ = server.join();
    }
}
