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

//! Copying one file over SFTP. A file arrives under a temporary name, its size
//! is checked, and only then does it take its real name, so a broken or
//! cancelled copy never leaves half a file where the real one belongs.

use anyhow::{Context, Result, anyhow};
use russh_sftp::client::SftpSession;
use russh_sftp::protocol::{FileAttributes, OpenFlags};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncWriteExt, BufReader};

/// The read size a copy streams with: large enough to fill SFTP packets, and
/// the most a file of any size holds in memory.
const COPY_BUFFER: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Upload,
    Download,
}

impl std::str::FromStr for Direction {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "upload" => Ok(Self::Upload),
            "download" => Ok(Self::Download),
            other => Err(anyhow!(
                "direction must be 'upload' or 'download', not '{other}'"
            )),
        }
    }
}

/// A remote temporary file, removed unless it was renamed into place. Dropped
/// mid-copy when the user cancels the turn.
struct RemotePart {
    sftp: Arc<SftpSession>,
    path: Option<String>,
}

impl Drop for RemotePart {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let sftp = self.sftp.clone();
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    let _ = sftp.remove_file(path).await;
                });
            }
        }
    }
}

/// A local temporary file, removed unless it was renamed into place.
struct LocalPart(Option<PathBuf>);

impl Drop for LocalPart {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// SFTP paths start at the home directory when relative, which is what `~`
/// would mean to a shell; the server does not expand it.
fn remote_path(raw: &str) -> Result<String> {
    if raw.is_empty() {
        return Err(anyhow!("'remote' is empty"));
    }
    Ok(match raw {
        "~" | "~/" => ".".to_string(),
        _ => raw.strip_prefix("~/").unwrap_or(raw).to_string(),
    })
}

fn join_remote(directory: &str, name: &str) -> String {
    if directory.ends_with('/') {
        format!("{directory}{name}")
    } else {
        format!("{directory}/{name}")
    }
}

/// The last component of a remote path, which comes from the server and
/// becomes a local file name.
fn remote_file_name(path: &str) -> Result<&str> {
    match path.trim_end_matches('/').rsplit('/').next() {
        Some(name) if !name.is_empty() && name != "." && name != ".." && !name.contains('\\') => {
            Ok(name)
        }
        _ => Err(anyhow!(
            "cannot name a local file after remote path '{path}'"
        )),
    }
}

fn part_name() -> String {
    format!(".orangu-part-{}", uuid::Uuid::new_v4().simple())
}

fn report(verb: &str, from: &str, to: &str, bytes: u64, started: std::time::Instant) -> String {
    let seconds = started.elapsed().as_secs_f64();
    format!(
        "{verb} {from} to {to}: {bytes} bytes in {seconds:.1} s ({:.1} MiB/s).",
        bytes as f64 / 1_048_576.0 / seconds.max(0.001)
    )
}

pub async fn copy(
    sftp: &Arc<SftpSession>,
    direction: Direction,
    workspace: &Path,
    local: &str,
    remote: &str,
    overwrite: bool,
    label: &str,
) -> Result<String> {
    let remote = remote_path(remote)?;
    match direction {
        Direction::Upload => upload(sftp, workspace, local, &remote, overwrite, label).await,
        Direction::Download => download(sftp, workspace, local, &remote, overwrite, label).await,
    }
}

async fn upload(
    sftp: &Arc<SftpSession>,
    workspace: &Path,
    local: &str,
    remote: &str,
    overwrite: bool,
    label: &str,
) -> Result<String> {
    let source = crate::tools::resolve_workspace_path(workspace, local)?;
    let metadata = tokio::fs::metadata(&source)
        .await
        .with_context(|| format!("reading {local}"))?;
    if !metadata.is_file() {
        return Err(anyhow!(
            "{local} is not a regular file; ssh_copy copies one file at a time"
        ));
    }
    let name = source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("{local} has no usable file name"))?;

    let into_directory = remote.ends_with('/')
        || sftp
            .metadata(remote.to_string())
            .await
            .is_ok_and(|attributes| attributes.is_dir());
    let target = if into_directory {
        join_remote(remote, name)
    } else {
        remote.to_string()
    };
    let exists = sftp.try_exists(target.clone()).await.unwrap_or(false);
    if exists && !overwrite {
        return Err(anyhow!(
            "{target} already exists on {label}; pass overwrite to replace it"
        ));
    }

    let mut attributes = FileAttributes::empty();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        attributes.permissions = Some(metadata.permissions().mode() & 0o777);
    }
    let part_path = match target.rsplit_once('/') {
        Some((directory, _)) => format!("{directory}/{}", part_name()),
        None => part_name(),
    };
    let started = std::time::Instant::now();
    let mut remote_file = sftp
        .open_with_flags_and_attributes(
            part_path.clone(),
            OpenFlags::CREATE | OpenFlags::EXCLUDE | OpenFlags::WRITE,
            attributes,
        )
        .await
        .with_context(|| format!("creating a file next to {target} on {label}"))?;
    let mut part = RemotePart {
        sftp: sftp.clone(),
        path: Some(part_path.clone()),
    };
    let file = tokio::fs::File::open(&source)
        .await
        .with_context(|| format!("opening {local}"))?;
    let mut reader = BufReader::with_capacity(COPY_BUFFER, file);
    let bytes = tokio::io::copy_buf(&mut reader, &mut remote_file)
        .await
        .with_context(|| format!("copying {local} to {label}"))?;
    remote_file
        .shutdown()
        .await
        .with_context(|| format!("finishing {target} on {label}"))?;

    let written = sftp
        .metadata(part_path.clone())
        .await
        .with_context(|| format!("checking {target} on {label}"))?
        .len();
    if written != bytes {
        return Err(anyhow!(
            "{label} holds {written} of the {bytes} bytes sent; {target} was left unchanged"
        ));
    }
    // An SFTP rename does not replace an existing file.
    if exists {
        sftp.remove_file(target.clone())
            .await
            .with_context(|| format!("replacing {target} on {label}"))?;
    }
    sftp.rename(part_path, target.clone())
        .await
        .with_context(|| format!("naming {target} on {label}"))?;
    part.path = None;
    Ok(report(
        "Uploaded",
        local,
        &format!("{label}:{target}"),
        bytes,
        started,
    ))
}

async fn download(
    sftp: &Arc<SftpSession>,
    workspace: &Path,
    local: &str,
    remote: &str,
    overwrite: bool,
    label: &str,
) -> Result<String> {
    let attributes = sftp
        .metadata(remote.to_string())
        .await
        .with_context(|| format!("reading {remote} on {label}"))?;
    if attributes.is_dir() {
        return Err(anyhow!(
            "{remote} on {label} is a directory; ssh_copy copies one file at a time"
        ));
    }
    if attributes.permissions.is_some() && !attributes.is_regular() {
        return Err(anyhow!("{remote} on {label} is not a regular file"));
    }

    let mut target = crate::tools::resolve_workspace_path(workspace, local)?;
    if local.ends_with('/') || target.is_dir() {
        target = crate::tools::resolve_workspace_path(
            workspace,
            &target.join(remote_file_name(remote)?).to_string_lossy(),
        )?;
    }
    if target.exists() && !overwrite {
        return Err(anyhow!(
            "{} already exists; pass overwrite to replace it",
            target.display()
        ));
    }
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let part_path = target.with_file_name(part_name());

    let started = std::time::Instant::now();
    let remote_file = sftp
        .open(remote.to_string())
        .await
        .with_context(|| format!("opening {remote} on {label}"))?;
    // Private until complete, whatever the file's mode ends up being.
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut part_file = options
        .open(&part_path)
        .await
        .with_context(|| format!("creating a file next to {}", target.display()))?;
    let mut part = LocalPart(Some(part_path.clone()));
    let mut reader = BufReader::with_capacity(COPY_BUFFER, remote_file);
    let bytes = tokio::io::copy_buf(&mut reader, &mut part_file)
        .await
        .with_context(|| format!("copying {remote} from {label}"))?;
    part_file.flush().await?;
    part_file.sync_all().await?;
    drop(part_file);

    let expected = attributes.len();
    if attributes.size.is_some() && bytes != expected {
        return Err(anyhow!(
            "received {bytes} of the {expected} bytes {label} reports for {remote}; {} was left \
             unchanged",
            target.display()
        ));
    }
    #[cfg(unix)]
    if let Some(mode) = attributes.permissions {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(&part_path, std::fs::Permissions::from_mode(mode & 0o777))
            .await?;
    }
    tokio::fs::rename(&part_path, &target)
        .await
        .with_context(|| format!("naming {}", target.display()))?;
    part.0 = None;
    Ok(report(
        "Downloaded",
        &format!("{label}:{remote}"),
        &target.display().to_string(),
        bytes,
        started,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tilde_means_the_home_directory() {
        assert_eq!(remote_path("~/a/b").unwrap(), "a/b");
        assert_eq!(remote_path("~").unwrap(), ".");
        assert_eq!(remote_path("/etc/hosts").unwrap(), "/etc/hosts");
        assert!(remote_path("").is_err());
    }

    /// A remote name becomes a local path, so it may not climb out of the
    /// directory it lands in.
    #[test]
    fn a_remote_name_cannot_escape_the_local_directory() {
        assert_eq!(remote_file_name("/a/b.txt").unwrap(), "b.txt");
        assert_eq!(remote_file_name("/a/b/").unwrap(), "b");
        for path in ["/", "..", "/a/..", ".", "a\\..\\b"] {
            assert!(remote_file_name(path).is_err(), "{path}");
        }
    }
}
