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

//! Spreading one model's layers over a tree of `orangu-server` nodes —
//! the `[workers]` section.
//!
//! - [`protocol`]: the frames a parent and a worker exchange.
//! - [`auth`]: proving `[workers].secret` without sending it.
//! - [`channel`]: finding the other nodes of `[workers].name`.
//! - [`identity`]: making sure every node has the same model, quantization
//!   included.
//! - [`pipeline`]: one node's slice — its own layers, then its workers' —
//!   and the top-level node's [`pipeline::DelegatingModel`].
//! - [`stage`]: the parent side of a forward; [`link`]: its TCP connection.
//! - [`session`]: the worker side of a forward.
//! - [`plan`]: dividing a range of layers by memory.
//! - [`prefixes`]: a tree's prefix pool, saved per node.
//! - [`prefill`]: where a prompt runs — through the tree, or on one node
//!   that holds the whole model, its rows then handed to every node.
//! - [`node`]: one process's place in the tree, both halves.
//! - [`divide`]: whether a request runs on one node alone or divided over a
//!   tree, from what each node measured.
//! - [`syncbench`]: timing the sums two nodes splitting every layer would
//!   exchange, against a peer of the channel.
//! - [`load`]: how busy a node is, for routing requests and sizing shares.
//! - [`demand`]: requests for a model no free node of the channel served.
//! - [`clock`]: keeping a node's cores at full clock while a request runs.
//! - [`transport`]: plain TCP or TLS between nodes.
//! - [`fetch`]: a node that downloads only what it runs; [`share`]: byte
//!   ranges of a node's model file, served to such a node.

pub use orangu::workers::{auth, channel, protocol, transport};
pub mod clock;
pub mod demand;
pub mod divide;
pub mod fetch;
pub mod guard;
pub mod identity;
pub mod link;
pub mod load;
pub mod metrics;
pub mod node;
pub mod pipeline;
pub mod plan;
pub mod prefill;
pub mod prefixes;
pub mod room;
pub mod session;
pub mod share;
pub mod speed;
pub mod stage;
pub mod syncbench;

#[cfg(test)]
pub mod fixture {
    use std::path::{Path, PathBuf};

    /// A self-signed certificate for `127.0.0.1` and `localhost`, and its
    /// key, written under `dir` with `openssl` as the TLS tests do.
    pub fn certificate(dir: &Path, name: &str) -> (PathBuf, PathBuf) {
        let cert = dir.join(format!("{name}.pem"));
        let key = dir.join(format!("{name}.key"));
        let out = std::process::Command::new("openssl")
            .args([
                "req",
                "-x509",
                "-newkey",
                "ec",
                "-pkeyopt",
                "ec_paramgen_curve:prime256v1",
                "-nodes",
                "-days",
                "1",
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=IP:127.0.0.1,DNS:localhost",
                // A server's own certificate, not an authority: rustls
                // refuses a CA certificate presented as a server's.
                "-addext",
                "basicConstraints=critical,CA:FALSE",
                "-keyout",
                key.to_str().unwrap(),
                "-out",
                cert.to_str().unwrap(),
            ])
            .output()
            .expect("openssl is needed for this test");
        assert!(
            out.status.success(),
            "openssl: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        (cert, key)
    }
}
