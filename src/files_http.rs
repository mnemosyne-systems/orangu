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

//! The file-lifecycle API as a mountable axum router: `POST
//! /v1/create_file`, `/v1/modify_file`, `/v1/move_file`, `/v1/delete_file`,
//! `/v1/show_file`, `/v1/create_directory`, `/v1/move_directory`,
//! `/v1/delete_directory`.
//!
//! Mounted by `orangu-server` beside its inference endpoints. It is the
//! server that owns a workspace and does the work; `orangu-coordinator` is
//! an HTTP proxy in front of it and simply forwards these requests, like
//! every other request it sees, to the backend it has active.
//!
//! The state a host passes only has to name its workspace root, via
//! [`WorkspaceState`]. The router is factored out here, rather than left in
//! the server binary, so the operations ([`crate::files`]) and their HTTP
//! shape stay one thing that `orangu`'s own tools and typed commands share.

use axum::{
    Json, Router,
    extract::{FromRequest, Request, State, rejection::JsonRejection},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
};
use serde::de::DeserializeOwned;
use std::path::Path;
use std::sync::Arc;

use crate::files::{
    self, CreateDirectoryRequest, CreateFileRequest, DeleteDirectoryRequest, DeleteFileRequest,
    FileError, ModifyFileRequest, MoveDirectoryRequest, MoveFileRequest, ShowFileRequest,
};

/// What these endpoints need from whatever server hosts them: the workspace
/// root every request is resolved against and confined to.
pub trait WorkspaceState: Send + Sync + 'static {
    fn workspace(&self) -> &Path;
}

/// The eight endpoints, ready to `merge` into a host's own router.
pub fn router<S: WorkspaceState>() -> Router<Arc<S>> {
    Router::new()
        .route("/v1/create_file", post(create_file::<S>))
        .route("/v1/modify_file", post(modify_file::<S>))
        .route("/v1/move_file", post(move_file::<S>))
        .route("/v1/delete_file", post(delete_file::<S>))
        .route("/v1/show_file", post(show_file::<S>))
        .route("/v1/create_directory", post(create_directory::<S>))
        .route("/v1/move_directory", post(move_directory::<S>))
        .route("/v1/delete_directory", post(delete_directory::<S>))
}

/// The HTTP status one [`FileError`] answers with. The `code` string a
/// client branches on is the error's own (`FileError::code`); this is just
/// how each variant surfaces over HTTP.
pub fn status_for(error: &FileError) -> StatusCode {
    match error {
        FileError::OutsideWorkspace(_) => StatusCode::FORBIDDEN,
        FileError::NotFound(_) => StatusCode::NOT_FOUND,
        FileError::AlreadyExists(_) | FileError::NotEmpty(_) => StatusCode::CONFLICT,
        FileError::NotAFile(_)
        | FileError::NotADirectory(_)
        | FileError::BadRequest(_)
        | FileError::NotUtf8(_) => StatusCode::BAD_REQUEST,
        FileError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Wraps a [`FileError`] as a response body — `{"error": {"code",
/// "message"}}`, the one shape every failure here uses.
pub struct FileErrorResponse(FileError);

impl From<FileError> for FileErrorResponse {
    fn from(error: FileError) -> Self {
        Self(error)
    }
}

impl IntoResponse for FileErrorResponse {
    fn into_response(self) -> axum::response::Response {
        (
            status_for(&self.0),
            Json(serde_json::json!({
                "error": {
                    "code": self.0.code(),
                    "message": self.0.message(),
                }
            })),
        )
            .into_response()
    }
}

/// `Json<T>`, but rejecting a malformed body as a [`FileError`] so a client
/// that mistypes a field name gets the same `{"error": {"code", "message"}}`
/// shape (and a `400`) as every other failure here, instead of axum's own
/// plain-text `422`.
pub struct FileJson<T>(pub T);

impl<T, S> FromRequest<S> for FileJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = FileErrorResponse;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) =
            Json::<T>::from_request(req, state)
                .await
                .map_err(|rejection: JsonRejection| {
                    FileErrorResponse(FileError::BadRequest(rejection.body_text()))
                })?;
        Ok(FileJson(value))
    }
}

/// One handler per endpoint, each the same three steps: take the host's
/// workspace root, call the library operation, and let `?` turn a
/// [`FileError`] into its JSON body.
macro_rules! endpoint {
    ($handler:ident, $operation:path, $request:ty) => {
        async fn $handler<S: WorkspaceState>(
            State(state): State<Arc<S>>,
            FileJson(request): FileJson<$request>,
        ) -> Result<impl IntoResponse, FileErrorResponse> {
            Ok(Json($operation(state.workspace(), request)?))
        }
    };
}

endpoint!(create_file, files::create, CreateFileRequest);
endpoint!(modify_file, files::modify, ModifyFileRequest);
endpoint!(move_file, files::move_, MoveFileRequest);
endpoint!(delete_file, files::delete, DeleteFileRequest);
endpoint!(show_file, files::show, ShowFileRequest);
endpoint!(create_directory, files::create_dir, CreateDirectoryRequest);
endpoint!(move_directory, files::move_dir, MoveDirectoryRequest);
endpoint!(delete_directory, files::delete_dir, DeleteDirectoryRequest);

/// Every path this router serves — the canonical list, kept in step with
/// the routes above by this module's own test. For anything that needs to
/// name the file-lifecycle endpoints as a set (documentation, logging, a
/// proxy's own routing table) rather than re-deriving them.
pub const PATHS: &[&str] = &[
    "/v1/create_file",
    "/v1/modify_file",
    "/v1/move_file",
    "/v1/delete_file",
    "/v1/show_file",
    "/v1/create_directory",
    "/v1/move_directory",
    "/v1/delete_directory",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The status mapping is this module's own job; the operations
    /// themselves are covered in `crate::files`.
    #[test]
    fn each_error_maps_to_its_documented_status() {
        let cases = [
            (
                FileError::OutsideWorkspace(String::new()),
                StatusCode::FORBIDDEN,
                "outside_workspace",
            ),
            (
                FileError::NotFound(String::new()),
                StatusCode::NOT_FOUND,
                "not_found",
            ),
            (
                FileError::AlreadyExists(String::new()),
                StatusCode::CONFLICT,
                "already_exists",
            ),
            (
                FileError::NotEmpty(String::new()),
                StatusCode::CONFLICT,
                "not_empty",
            ),
            (
                FileError::NotAFile(String::new()),
                StatusCode::BAD_REQUEST,
                "not_a_file",
            ),
            (
                FileError::NotADirectory(String::new()),
                StatusCode::BAD_REQUEST,
                "not_a_directory",
            ),
            (
                FileError::BadRequest(String::new()),
                StatusCode::BAD_REQUEST,
                "bad_request",
            ),
            (
                FileError::NotUtf8(String::new()),
                StatusCode::BAD_REQUEST,
                "not_utf8",
            ),
            (
                FileError::Io(String::new()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
            ),
        ];

        for (error, status, code) in cases {
            assert_eq!(error.code(), code);
            assert_eq!(status_for(&error), status, "{code}");
        }
    }

    /// `PATHS` is what a host advertises as locally served, so it must
    /// stay in step with the routes actually registered above.
    #[test]
    fn paths_lists_every_route_the_router_registers() {
        let source = include_str!("files_http.rs");
        let registered: Vec<&str> = source
            .lines()
            .filter_map(|line| line.trim().strip_prefix(".route(\""))
            .filter_map(|rest| rest.split('"').next())
            .collect();

        assert_eq!(registered.len(), 8, "registered: {registered:?}");
        for path in PATHS {
            assert!(
                registered.contains(path),
                "PATHS lists {path}, router does not"
            );
        }
        for path in registered {
            assert!(
                PATHS.contains(&path),
                "router serves {path}, PATHS does not"
            );
        }
    }
}
