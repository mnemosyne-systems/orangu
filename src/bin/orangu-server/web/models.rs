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

//! The web console's model manager, on the same `web` port the chat UI is
//! served from: `orangu-server list` as the view, and `show`, `download` and
//! `delete` as the things that can be done from it.
//!
//! Every endpoint is the same operation the matching subcommand performs,
//! calling the *same* shared code rather than a second implementation of it:
//! `orangu::model_spec` for the directory scan and shard grouping (`list`),
//! down to `ModelSupport::cell` rendering the `SUPPORTED` column, so the
//! panel's table and the CLI's cannot say different things about the same
//! file; `crate::format_show` for the metadata dump (`show`);
//! `orangu::model_download` for the fetch (`download`); and
//! `orangu::model_spec::delete_model` for the removal (`delete`).
//!
//! **A download runs detached, not inside a request.** It takes minutes to
//! hours, far past any browser's patience: `POST` starts a [`Job`] on a
//! blocking thread and returns immediately, and the UI polls
//! `GET /api/models` for its progress. Only one runs at a time — two
//! concurrent fetches into one models directory would compete for the same
//! disk and the same free-space check.
//!
//! **Access is exactly the chat UI's.** These endpoints are neither
//! authenticated nor loopback-restricted, matching the rest of the `web`
//! port (and the file-lifecycle API on the API port) — the whole server
//! assumes a trusted network. A server reachable from an untrusted one
//! should not have `web` enabled at all.

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use super::WebState;
use orangu::model_download::{DownloadProgress, DownloadSnapshot};
use orangu::model_spec::ModelGroup;

pub fn router() -> Router<Arc<WebState>> {
    Router::new()
        .route("/api/models", get(list).delete(remove))
        .route("/api/models/metadata", get(metadata))
        .route("/api/models/select", post(select))
        .route("/api/models/download", post(download))
        .route("/api/models/updates", get(updates))
        .route("/api/models/job", delete(dismiss_job))
}

/// A download running on a blocking thread, and everything the UI needs to
/// draw its progress. Immutable except for the two interior cells that the
/// worker writes and the poller reads.
pub struct Job {
    /// The `<user>/<model>[:quant]` spec being fetched.
    pub spec: String,
    /// Filled in as the fetch runs; see [`DownloadProgress`].
    pub progress: Arc<DownloadProgress>,
    pub outcome: Mutex<Outcome>,
}

#[derive(Clone)]
pub enum Outcome {
    Running,
    /// Finished; the message names what landed where.
    Done(String),
    /// Failed; the message is the full `anyhow` chain.
    Failed(String),
}

/// The one job slot. A second `POST` while one is running is refused rather
/// than queued: the caller is a person looking at a page, and "already
/// downloading X" is a more useful answer than a silent wait.
#[derive(Default)]
pub struct ModelJobs {
    current: Mutex<Option<Arc<Job>>>,
}

impl ModelJobs {
    /// Claims the slot for a new job, or returns the spec of the one already
    /// holding it. A *finished* job doesn't hold the slot — its result stays
    /// readable until the next job replaces it or the UI dismisses it, so a
    /// completed download's "done" message survives a page refresh.
    fn claim(&self, spec: String) -> Result<Arc<Job>, String> {
        let mut current = self.current.lock().unwrap();
        if let Some(running) = current.as_ref()
            && matches!(*running.outcome.lock().unwrap(), Outcome::Running)
        {
            return Err(running.spec.clone());
        }
        let job = Arc::new(Job {
            spec,
            progress: Arc::new(DownloadProgress::default()),
            outcome: Mutex::new(Outcome::Running),
        });
        *current = Some(job.clone());
        Ok(job)
    }

    fn snapshot(&self) -> Option<JobView> {
        let job = self.current.lock().unwrap().clone()?;
        let (state, message) = match &*job.outcome.lock().unwrap() {
            Outcome::Running => ("running", None),
            Outcome::Done(msg) => ("done", Some(msg.clone())),
            Outcome::Failed(msg) => ("failed", Some(msg.clone())),
        };
        Some(JobView {
            spec: job.spec.clone(),
            state,
            message,
            progress: job.progress.snapshot(),
        })
    }

    /// Drops a *finished* job's result. A running one is left alone — there
    /// is no cancellation here, and pretending otherwise would leave the UI
    /// showing nothing while bytes kept landing on disk.
    fn dismiss(&self) {
        let mut current = self.current.lock().unwrap();
        let finished = current
            .as_ref()
            .is_some_and(|job| !matches!(*job.outcome.lock().unwrap(), Outcome::Running));
        if finished {
            *current = None;
        }
    }
}

#[derive(Serialize)]
pub struct JobView {
    spec: String,
    state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    progress: DownloadSnapshot,
}

/// The models-directory scan, kept so the panel can poll.
///
/// Building it opens the GGUF header of every model under the directory —
/// and then, via [`crate::model_support`], of every *shard* of every model
/// again. On a directory holding a few dozen multi-shard models that is
/// seconds of disk work, which is fine once and impossible once a second:
/// the panel polls for a download's progress, and that progress lives in
/// memory, not on disk.
///
/// So the scan is explicit rather than periodic. It is rebuilt when the
/// panel opens, after anything that changes the directory (a delete, a
/// finished download or refresh), and whenever the Rescan button asks —
/// which is also the answer for a `.gguf` copied in by hand, since nothing
/// else would notice that.
#[derive(Default)]
pub struct ModelCatalog {
    cached: Mutex<Option<Vec<CatalogEntry>>>,
}

/// One scanned model: the row as it will be serialized, plus the shard paths
/// [`ModelView::loaded`] is decided against — which depends on what is loaded
/// *now*, not on what was loaded when the scan ran, and so is filled in per
/// request rather than cached.
#[derive(Clone)]
struct CatalogEntry {
    view: ModelView,
    paths: Vec<PathBuf>,
}

impl ModelCatalog {
    /// Discards the cached scan, so the next [`Self::rows`] rebuilds it.
    fn invalidate(&self) {
        *self.cached.lock().unwrap() = None;
    }

    /// The scanned rows, rebuilding the cache if `rescan` or if there is
    /// none. Blocking — the caller runs it off the async worker threads.
    fn rows(
        &self,
        models_dir: &std::path::Path,
        rescan: bool,
    ) -> anyhow::Result<Vec<CatalogEntry>> {
        if !rescan && let Some(cached) = self.cached.lock().unwrap().as_ref() {
            return Ok(cached.clone());
        }
        let groups = scan(models_dir)?;
        let support = crate::model_support(&groups);
        let entries: Vec<CatalogEntry> = groups
            .iter()
            .zip(&support)
            .enumerate()
            .map(|(index, (group, support))| CatalogEntry {
                view: ModelView {
                    nr: index + 1,
                    label: group.label.clone(),
                    quant: group
                        .quantization
                        .clone()
                        .unwrap_or_else(|| "-".to_string()),
                    size: orangu::format::format_bytes(group.size_bytes),
                    supported: support.cell(),
                    loadable: support.loadable(),
                    // An error row carries neither SIZE nor SUPPORTED in the
                    // CLI's own table; the panel drops the same cells.
                    error: (!group.errors.is_empty()).then(|| group.errors.join("; ")),
                    refresh: false,
                    loaded: false,
                    path: group.representative_path.display().to_string(),
                },
                paths: group.paths.clone(),
            })
            .collect();
        *self.cached.lock().unwrap() = Some(entries.clone());
        Ok(entries)
    }
}

/// One row of `orangu-server list`, column for column: `NR`, `MODEL`,
/// `QUANT`, `SIZE`, `SUPPORTED`. The strings are the ones the CLI itself
/// would print — `quant` falls back to `-` the same way, `supported` is
/// [`ModelSupport::cell`] verbatim — so the panel's table and the terminal's
/// cannot end up saying different things about the same file.
#[derive(Clone, Serialize)]
struct ModelView {
    /// `NR`. The same number the CLI prints for the same model, being
    /// derived from the same scan in the same order.
    nr: usize,
    /// `MODEL`.
    label: String,
    /// `QUANT`, `-` when the file says nothing about its scheme.
    quant: String,
    /// `SIZE`, already formatted (`orangu::format::format_bytes`).
    size: String,
    /// `SUPPORTED`, e.g. `Yes (llama)` or `No (llama, TQ1_0)`.
    supported: String,
    /// Whether this build could actually load it. Not a column — it is what
    /// the CLI *greys* an unsupported row for, and the panel does the same.
    loadable: bool,
    /// The whole row replaced by `error: ...`, exactly as `list` does for a
    /// file whose header wouldn't parse.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    /// `list`'s `(Refresh)` marker: this model's repo has a newer revision
    /// on the Hub. Filled in by [`updates`], not by the scan.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    refresh: bool,
    /// Set for the model this server has loaded. Not a column either — it is
    /// why the row's Delete button is disabled.
    loaded: bool,
    /// The representative shard's path: the row's tooltip, and what a delete
    /// is checked against (see [`ModelRequest::path`]).
    path: String,
}

/// The model this server has loaded — the header line above the table,
/// saying which of the listed rows is actually answering requests.
#[derive(Serialize)]
struct CurrentView {
    display: String,
    architecture: String,
    backend: String,
    path: String,
    n_layer: usize,
    n_ctx: usize,
    role: &'static str,
    slots: usize,
}

#[derive(Serialize)]
struct ListView {
    models_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    disk_used_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    disk_available_bytes: Option<u64>,
    current: CurrentView,
    /// Whether **Load** is offered at all: `[web].reexec` is on *and* this
    /// platform can `execve`. The panel disables the button when this is
    /// false rather than letting it fail on click.
    can_load: bool,
    /// `[web].delete`. The panel draws no Delete button at all when this is
    /// false — see [`WebState::can_delete`].
    can_delete: bool,
    /// Set once a handover has been accepted — the process is about to be
    /// replaced. The panel shows it and stops offering more actions; its
    /// next poll is the one that lands on the new image.
    #[serde(skip_serializing_if = "Option::is_none")]
    loading: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    job: Option<JobView>,
    models: Vec<ModelView>,
}

#[derive(Deserialize)]
struct ListQuery {
    /// Re-read the models directory instead of serving the cached scan —
    /// see [`ModelCatalog`]. The panel sets this when it opens, after an
    /// action, and from its Rescan button; its once-a-second poll does not.
    #[serde(default)]
    rescan: bool,
}

/// Everything the manager panel draws in one response: `orangu-server
/// list`'s own table, which of its rows this server has loaded, where the
/// models live, and whatever download is running. Deliberately does *not*
/// contact the Hugging Face Hub — see [`updates`].
async fn list(
    State(state): State<Arc<WebState>>,
    Query(query): Query<ListQuery>,
) -> impl IntoResponse {
    let models_dir = state.models_dir.clone();
    let catalog = state.catalog.clone();
    // Cached or not, the scan behind this is disk work — off the async
    // runtime's worker threads either way.
    let scanned = tokio::task::spawn_blocking(move || {
        let rows = catalog.rows(&models_dir, query.rescan)?;
        let storage = orangu::os::detect_model_storage(&models_dir);
        anyhow::Ok((rows, storage))
    })
    .await;

    let (rows, storage) = match scanned {
        Ok(Ok(scanned)) => scanned,
        Ok(Err(err)) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")).into_response();
        }
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };

    let cfg = state.engine.model.config();
    Json(ListView {
        models_dir: state.models_dir.display().to_string(),
        disk_used_bytes: storage.as_ref().map(|s| s.used_bytes),
        disk_available_bytes: storage.as_ref().and_then(|s| s.available_bytes),
        current: CurrentView {
            display: state.model_display.clone(),
            architecture: state.architecture.clone(),
            backend: state.backend_label.clone(),
            path: state.model_path.display().to_string(),
            n_layer: cfg.n_layer,
            n_ctx: cfg.n_ctx_train,
            role: state.engine.role.label(),
            slots: state.engine.slots.total(),
        },
        can_load: state.handover.is_some(),
        can_delete: state.can_delete,
        loading: state.loading_model(),
        job: state.jobs.snapshot(),
        // Which row is loaded is decided here rather than baked into the
        // cached scan: the scan is about the directory, this is about this
        // process.
        models: rows
            .into_iter()
            .map(|entry| ModelView {
                loaded: entry.paths.iter().any(|path| *path == state.model_path),
                ..entry.view
            })
            .collect(),
    })
    .into_response()
}

/// `list`'s own `(Refresh)` marker: which rows have moved on in their
/// Hugging Face repo, by `NR`. One Hub lookup per distinct repo, via the
/// same [`crate::check_for_updates`] the CLI table uses.
///
/// Its own endpoint rather than part of [`list`] because it is a network
/// round trip per repo: the panel opens instantly on local state and marks
/// the rows when this answers — or never, on a machine with no internet,
/// which is exactly the "unknown, so not behind" the CLI already treats it
/// as. `orangu-server refresh` is what acts on a marked row.
async fn updates(State(state): State<Arc<WebState>>) -> impl IntoResponse {
    let models_dir = state.models_dir.clone();
    let behind = tokio::task::spawn_blocking(move || {
        // Its own scan, not the catalog's cached rows: `is_behind` needs each
        // group's `local_commit`, which the rows don't carry. Only ever run
        // when the panel opens or the Rescan button is pressed, never on the
        // poll — it is a Hub round trip per repo on top of the scan.
        let groups = scan(&models_dir)?;
        let latest = crate::check_for_updates(&groups);
        anyhow::Ok(
            groups
                .iter()
                .enumerate()
                .filter(|(_, group)| group.is_behind(&latest))
                .map(|(index, _)| index + 1)
                .collect::<Vec<usize>>(),
        )
    })
    .await;

    match behind {
        Ok(Ok(behind)) => Json(serde_json::json!({ "behind": behind })).into_response(),
        Ok(Err(err)) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
struct MetadataQuery {
    /// A `MODEL` label, an `NR`, a bare filename, or a path — resolved by
    /// [`orangu::model_spec::resolve_show_target`], exactly as
    /// `orangu-server show`'s own argument is.
    model: String,
    /// Also list every tensor's name, shape, type and offset — `show
    /// --tensors`.
    #[serde(default)]
    tensors: bool,
    /// Print every array element instead of a truncated preview — `show
    /// --full`. A vocabulary is 100,000+ entries, so this is off by default
    /// here for the same reason it is on the CLI.
    #[serde(default)]
    full: bool,
}

/// `orangu-server show` as text, for the panel's metadata viewer — the same
/// [`crate::format_show`] output the CLI prints, not a second rendering of
/// the same metadata that could drift from it.
async fn metadata(
    State(state): State<Arc<WebState>>,
    Query(query): Query<MetadataQuery>,
) -> impl IntoResponse {
    let models_dir = state.models_dir.clone();
    let rendered = tokio::task::spawn_blocking(move || {
        let path = orangu::model_spec::resolve_show_target(&models_dir, &query.model)?;
        let gguf = orangu::gguf::GgufFile::open(&path)?;
        anyhow::Ok(format!(
            "{}\n{}",
            path.display(),
            crate::format_show(&gguf, query.full, query.tensors)
        ))
    })
    .await;

    match rendered {
        Ok(Ok(text)) => (
            StatusCode::OK,
            [
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Cache-Control", "no-store"),
            ],
            text,
        )
            .into_response(),
        Ok(Err(err)) => (StatusCode::NOT_FOUND, format!("{err:#}")).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
struct ModelRequest {
    /// A `MODEL` label from the listing, or an `NR`, or a path — whatever
    /// [`orangu::model_spec`]'s own resolution accepts. The UI sends the
    /// `NR`, which is the only spelling that names one row exactly: a repo
    /// with several quantizations on disk prints the same bare `MODEL` on
    /// each of their rows, and resolving that takes whichever comes first.
    model: String,
    /// The path the caller believed `model` resolves to — [`ModelView::path`]
    /// from the same listing the click came from. Checked before anything
    /// irreversible happens, because an `NR` is a *position*: a download
    /// finishing while a confirmation dialog is open re-sorts the listing
    /// underneath it, and row 20 is then a different model than the one
    /// named in the dialog the user just accepted. Optional, so a
    /// hand-written `curl` still works the way the CLI does.
    #[serde(default)]
    path: Option<String>,
}

impl ModelRequest {
    /// Refuses when the resolved group isn't the one the caller was looking
    /// at — see [`ModelRequest::path`].
    fn check_still_matches(&self, group: &ModelGroup) -> Result<(), String> {
        let Some(expected) = &self.path else {
            return Ok(());
        };
        if group.representative_path.to_string_lossy() == expected.as_str() {
            return Ok(());
        }
        Err(format!(
            "the model listing changed since this page drew it — '{}' is now '{}'. Reload and try \
             again.",
            self.model, group.label
        ))
    }
}

/// Loads a different model, by replacing this process with one serving it —
/// see `crate::reexec` for what that keeps (the listening sockets, the pid,
/// the detached state) and what it doesn't (anything in memory).
///
/// Everything that can be checked while the current model is still working
/// is checked here, before the handover is armed, because after the exec
/// there is no going back to it: that the config allows it at all, that no
/// slot is mid-generation, that the spec resolves, and that the file's
/// header names an architecture and quantization this build can read.
///
/// **Answers before it acts.** `execve` leaves no "after" to answer from, so
/// this returns `202` and arms the handover on a short timer — long enough
/// for the response to reach the client. That timer is best-effort UX, not
/// correctness: a client that gets a connection reset instead of the `202`
/// is looking at exactly the same thing happening, and its next poll lands
/// on the new image either way.
async fn select(
    State(state): State<Arc<WebState>>,
    Json(req): Json<ModelRequest>,
) -> impl IntoResponse {
    let Some(handover) = state.handover.clone() else {
        return (
            StatusCode::FORBIDDEN,
            format!(
                "loading a model from the web console is disabled ({})",
                if crate::reexec::supported() {
                    "set reexec = yes in orangu-server.conf to enable it"
                } else {
                    "this platform has no execve"
                }
            ),
        )
            .into_response();
    };

    // A generation in flight would be cut off mid-stream by the exec. Refused
    // rather than silently killed — the caller is a person who can wait for
    // their answer and click again.
    //
    // Not airtight, and cannot be: a request that has arrived but has not yet
    // acquired its slot is not yet `busy`, so one landing in that window is
    // still cut off. Closing it would mean a barrier between accepting
    // requests and running them, for a button a person presses — the cost is
    // one interrupted request in a race nobody can hit deliberately, and the
    // client simply retries.
    let busy = state
        .engine
        .slots
        .snapshot()
        .iter()
        .filter(|s| s.busy)
        .count();
    if busy > 0 {
        return (
            StatusCode::CONFLICT,
            format!("{busy} slot(s) still generating — try again once they finish"),
        )
            .into_response();
    }

    let models_dir = state.models_dir.clone();
    let loaded_path = state.model_path.clone();
    let resolved = tokio::task::spawn_blocking(move || {
        // `resolve_load_target`, not `resolve_or_fetch_model`: the label it
        // returns is what goes into `argv`, so an `NR` must not survive as
        // one, and a label two rows share must not either — see that
        // function's own doc comment.
        let (path, label) = orangu::model_spec::resolve_load_target(&models_dir, &req.model)?;
        // The same guard the delete path applies, and for a milder version
        // of the same reason: an `NR` is a position, and restarting the
        // server on a row the caller wasn't looking at is not destructive
        // but is certainly not what they asked for.
        if let Some(expected) = &req.path
            && path.to_string_lossy() != expected.as_str()
        {
            anyhow::bail!(
                "the model listing changed since this page drew it — '{}' is now '{label}'. \
                 Reload and try again.",
                req.model
            );
        }
        // The header check the `SUPPORTED` column already reports, run again
        // here against the file actually about to be loaded.
        crate::reexec::precheck(&path)?;
        anyhow::Ok((path, label))
    })
    .await;

    let (path, label) = match resolved {
        Ok(Ok(resolved)) => resolved,
        Ok(Err(err)) => return (StatusCode::BAD_REQUEST, format!("{err:#}")).into_response(),
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };

    if path == loaded_path {
        return (
            StatusCode::OK,
            Json(serde_json::json!({"model": label, "loaded": true})),
        )
            .into_response();
    }

    // One handover per process — there is only one process to replace, and a
    // second request arriving during the grace period below would exec into
    // a race with the first.
    if !state.arm_handover(&label) {
        return (
            StatusCode::CONFLICT,
            "a model is already being loaded".to_string(),
        )
            .into_response();
    }

    let previous = handover.current_model().to_string();
    let label_for_reply = label.clone();
    tokio::spawn(async move {
        tokio::time::sleep(HANDOVER_GRACE).await;
        // Only returns if the exec itself failed, which leaves this process
        // running the model it already had — worth saying out loud, since
        // the client has long since been told the handover was accepted.
        eprintln!(
            "error: {:#}",
            handover.exec(&label, Some(previous.as_str()))
        );
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({"loading": label_for_reply})),
    )
        .into_response()
}

/// How long the `202` from [`select`] gets to reach the client before the
/// process is replaced. Long enough for a loopback or LAN response to flush,
/// short enough not to read as a hang; see [`select`] for why missing it is
/// harmless.
const HANDOVER_GRACE: std::time::Duration = std::time::Duration::from_millis(300);

#[derive(Deserialize)]
struct DownloadRequest {
    /// A Hugging Face repo, `<user>/<model>[:quant]`. Without `:quant`, the
    /// same Q4_K_M-then-Q8_0 preference `orangu-server download` applies.
    repo: String,
}

/// Starts a Hugging Face download in the background and returns at once —
/// the UI watches `job` in [`list`]'s response for its progress.
async fn download(
    State(state): State<Arc<WebState>>,
    Json(req): Json<DownloadRequest>,
) -> impl IntoResponse {
    let repo = req.repo.trim().to_string();
    if repo.is_empty() {
        return (StatusCode::BAD_REQUEST, "no repo given").into_response();
    }
    let job = match state.jobs.claim(repo.clone()) {
        Ok(job) => job,
        Err(busy) => {
            return (StatusCode::CONFLICT, format!("already working on '{busy}'")).into_response();
        }
    };

    let models_dir = state.models_dir.clone();
    let catalog = state.catalog.clone();
    spawn_job(job.clone(), move |job| {
        let path = orangu::model_download::download_model_reporting(
            &models_dir,
            &repo,
            Some(job.progress.clone()),
        );
        // Whatever happened, the directory may have changed — a failure
        // partway through a multi-shard fetch still left files on disk.
        catalog.invalidate();
        Ok(format!("Downloaded to {}", path?.display()))
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({"spec": job.spec})),
    )
        .into_response()
}

/// Deletes every shard of a model, reclaiming its hub-cache blob(s) too when
/// nothing else still references them — `orangu-server delete`.
///
/// Refused outright when `[web].delete` is off — the panel draws no button
/// for it then, so reaching this at all means a hand-made request.
///
/// The loaded model is refused: its weights are memory-mapped by the running
/// engine, so deleting the file underneath it leaves this process reading a
/// file that no longer has a name and the next request generating from
/// whatever the kernel still has cached. Which model is served is a start-up
/// decision, so the way out is to restart on a different one — which the
/// message says, since the panel has no other way to change it.
async fn remove(
    State(state): State<Arc<WebState>>,
    Json(req): Json<ModelRequest>,
) -> impl IntoResponse {
    if !state.can_delete {
        return (
            StatusCode::FORBIDDEN,
            "deleting models from the web console is disabled (set delete = yes in the [web] \
             section of orangu-server.conf to enable it)",
        )
            .into_response();
    }

    let models_dir = state.models_dir.clone();
    let loaded_path = state.model_path.clone();
    let catalog = state.catalog.clone();
    let removed = tokio::task::spawn_blocking(move || {
        let group = orangu::model_spec::resolve_delete_target(&models_dir, &req.model)?;
        if let Err(err) = req.check_still_matches(&group) {
            anyhow::bail!("{err}");
        }
        if group.paths.contains(&loaded_path) {
            anyhow::bail!(
                "'{}' is the model this server is serving — its weights are mapped by the \
                 running engine. Restart on a different model to delete this one.",
                group.label
            );
        }
        let size = group.size_bytes;
        let files = group.paths.len();
        let label = group.label.clone();
        orangu::model_spec::delete_model(&models_dir, &group)?;
        catalog.invalidate();
        anyhow::Ok(format!(
            "Deleted '{label}' ({files} file{}, {})",
            if files == 1 { "" } else { "s" },
            orangu::format::format_bytes(size),
        ))
    })
    .await;

    match removed {
        Ok(Ok(message)) => Json(serde_json::json!({ "message": message })).into_response(),
        Ok(Err(err)) => (StatusCode::CONFLICT, format!("{err:#}")).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

async fn dismiss_job(State(state): State<Arc<WebState>>) -> impl IntoResponse {
    state.jobs.dismiss();
    StatusCode::NO_CONTENT
}

/// Runs `work` on a blocking thread and records what it returned on `job`.
/// Detached on purpose: the `POST` that started it has already answered, and
/// a browser tab closing must not abandon a download part-way.
fn spawn_job<F>(job: Arc<Job>, work: F)
where
    F: FnOnce(&Job) -> anyhow::Result<String> + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let outcome = match work(&job) {
            Ok(message) => Outcome::Done(message),
            Err(err) => Outcome::Failed(format!("{err:#}")),
        };
        *job.outcome.lock().unwrap() = outcome;
    });
}

/// The models directory as one group per model, the same scan-then-group
/// `orangu-server list` runs.
fn scan(models_dir: &std::path::Path) -> anyhow::Result<Vec<ModelGroup>> {
    let models = orangu::model_spec::scan_models_dir(models_dir)?;
    Ok(orangu::model_spec::group_models(&models))
}

#[cfg(test)]
mod tests {
    use super::*;
    use orangu::model_spec::ModelGroup;
    use std::io::Write;

    fn group(label: &str, representative: &str) -> ModelGroup {
        ModelGroup {
            label: label.to_string(),
            size_bytes: 0,
            quantization: None,
            errors: Vec::new(),
            representative_path: PathBuf::from(representative),
            paths: vec![PathBuf::from(representative)],
            hf_repo: None,
            local_commit: None,
        }
    }

    fn request(model: &str, path: Option<&str>) -> ModelRequest {
        ModelRequest {
            model: model.to_string(),
            path: path.map(str::to_string),
        }
    }

    /// The guard that stops an `NR` from deleting the wrong row: a click
    /// carries the path the listing showed, and a listing that has since been
    /// re-sorted resolves that same `NR` to a different model.
    #[test]
    fn a_stale_row_number_is_refused_rather_than_acted_on() {
        let intended = group("user/wanted", "/models/wanted.gguf");
        assert!(
            request("7", Some("/models/wanted.gguf"))
                .check_still_matches(&intended)
                .is_ok()
        );

        let now_something_else = group("user/other", "/models/other.gguf");
        let err = request("7", Some("/models/wanted.gguf"))
            .check_still_matches(&now_something_else)
            .unwrap_err();
        assert!(err.contains("listing changed"), "{err}");
        assert!(err.contains("user/other"), "{err}");
    }

    /// A caller that sends no path — `curl`, a script — gets the CLI's own
    /// behavior rather than a refusal.
    #[test]
    fn a_request_without_a_path_is_not_second_guessed() {
        assert!(
            request("user/anything", None)
                .check_still_matches(&group("user/anything", "/models/a.gguf"))
                .is_ok()
        );
    }

    /// One fetch at a time: a second `POST` while one is running is refused
    /// with the name of the one holding the slot, not silently queued.
    #[test]
    fn only_one_job_runs_at_a_time() {
        let jobs = ModelJobs::default();
        let first = jobs.claim("user/first".to_string()).unwrap();

        let busy = jobs
            .claim("user/second".to_string())
            .err()
            .expect("a second job must be refused while one is running");
        assert_eq!(busy, "user/first");

        // A finished job doesn't hold the slot — but its result is still
        // readable, so a completed download survives a page refresh.
        *first.outcome.lock().unwrap() = Outcome::Done("done".to_string());
        assert!(jobs.claim("user/second".to_string()).is_ok());
        assert_eq!(jobs.snapshot().unwrap().spec, "user/second");
    }

    /// Dismiss clears a result; it is not a cancel button in disguise.
    #[test]
    fn dismiss_drops_a_finished_job_and_leaves_a_running_one() {
        let jobs = ModelJobs::default();
        let job = jobs.claim("user/model".to_string()).unwrap();

        jobs.dismiss();
        assert!(
            jobs.snapshot().is_some(),
            "a running job must stay visible — nothing cancelled it"
        );

        *job.outcome.lock().unwrap() = Outcome::Failed("boom".to_string());
        jobs.dismiss();
        assert!(jobs.snapshot().is_none());
    }

    /// Writes a minimal GGUF — enough header for `scan_models_dir` to count
    /// it as a model.
    fn write_minimal_gguf(path: &std::path::Path, architecture: &str) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        buf.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
        buf.extend_from_slice(architecture.as_bytes());
        std::fs::File::create(path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    /// The panel polls once a second and the scan behind it opens every GGUF
    /// header under the models directory — so a poll must serve the cached
    /// scan, and only an explicit rescan (or something that changed the
    /// directory) may pay for a new one.
    #[test]
    fn the_catalog_is_only_re_read_when_asked() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_gguf(&dir.path().join("first.gguf"), "llama");

        let catalog = ModelCatalog::default();
        assert_eq!(catalog.rows(dir.path(), true).unwrap().len(), 1);

        write_minimal_gguf(&dir.path().join("second.gguf"), "llama");
        assert_eq!(
            catalog.rows(dir.path(), false).unwrap().len(),
            1,
            "a poll must not pay for a rescan"
        );
        assert_eq!(catalog.rows(dir.path(), true).unwrap().len(), 2);

        // What a delete or a finished download does instead of asking for a
        // rescan: the next read, poll or not, sees the new state.
        std::fs::remove_file(dir.path().join("second.gguf")).unwrap();
        catalog.invalidate();
        assert_eq!(catalog.rows(dir.path(), false).unwrap().len(), 1);
    }
}
