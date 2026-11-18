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

//! Mutual authentication of a parent and a worker from `[workers].secret`.
//!
//! The secret never crosses the wire. Each side sends a fresh random nonce;
//! the worker proves it knows the secret with an HMAC-SHA256 over both
//! nonces, and the parent answers with its own over the same two nonces in
//! the other order and under a different label, so neither proof can be
//! replayed as the other or into another connection:
//!
//! ```text
//! parent → Hello    { nonce_p }
//! worker → HelloAck { nonce_w, proof_w = HMAC(secret, "orangu-worker" ‖ nonce_p ‖ nonce_w) }
//! parent → Auth     { proof_p = HMAC(secret, "orangu-parent" ‖ nonce_w ‖ nonce_p) }
//! ```
//!
//! A side with no secret sends a proof of zeros and accepts whatever it is
//! sent; a side with a secret refuses a peer that cannot prove it. So a
//! secret on either end is enforced by that end.

use sha2::{Digest, Sha256};

const WORKER_LABEL: &[u8] = b"orangu-worker";
const PARENT_LABEL: &[u8] = b"orangu-parent";

/// HMAC-SHA256 (RFC 2104).
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut block = [0u8; BLOCK];
    if key.len() > BLOCK {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let pad = |byte: u8| -> [u8; BLOCK] { block.map(|k| k ^ byte) };
    let inner = Sha256::new()
        .chain_update(pad(0x36))
        .chain_update(message)
        .finalize();
    Sha256::new()
        .chain_update(pad(0x5c))
        .chain_update(inner)
        .finalize()
        .into()
}

/// A fresh random nonce for one handshake.
pub fn nonce() -> [u8; 32] {
    rand::random()
}

fn proof(secret: Option<&str>, label: &[u8], first: &[u8; 32], second: &[u8; 32]) -> [u8; 32] {
    match secret {
        Some(secret) => {
            let mut message = Vec::with_capacity(label.len() + 64);
            message.extend_from_slice(label);
            message.extend_from_slice(first);
            message.extend_from_slice(second);
            hmac_sha256(secret.as_bytes(), &message)
        }
        None => [0; 32],
    }
}

/// Compares in time independent of where the two differ.
fn same(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The worker's proof, sent in `HelloAck`.
pub fn worker_proof(
    secret: Option<&str>,
    parent_nonce: &[u8; 32],
    worker_nonce: &[u8; 32],
) -> [u8; 32] {
    proof(secret, WORKER_LABEL, parent_nonce, worker_nonce)
}

/// The parent's proof, sent in `Auth`.
pub fn parent_proof(
    secret: Option<&str>,
    parent_nonce: &[u8; 32],
    worker_nonce: &[u8; 32],
) -> [u8; 32] {
    proof(secret, PARENT_LABEL, worker_nonce, parent_nonce)
}

/// Whether the parent accepts the worker's `proof`.
pub fn parent_accepts(
    secret: Option<&str>,
    parent_nonce: &[u8; 32],
    worker_nonce: &[u8; 32],
    proof: &[u8; 32],
) -> bool {
    secret.is_none() || same(&worker_proof(secret, parent_nonce, worker_nonce), proof)
}

/// Whether the worker accepts the parent's `proof`.
pub fn worker_accepts(
    secret: Option<&str>,
    parent_nonce: &[u8; 32],
    worker_nonce: &[u8; 32],
    proof: &[u8; 32],
) -> bool {
    secret.is_none() || same(&parent_proof(secret, parent_nonce, worker_nonce), proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// RFC 4231, test cases 2 and 6 (a key longer than the block).
    #[test]
    fn hmac_matches_the_rfc_vectors() {
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn both_sides_accept_each_other_with_the_same_secret() {
        let (p, w) = (nonce(), nonce());
        let secret = Some("s3cret");
        assert!(parent_accepts(
            secret,
            &p,
            &w,
            &worker_proof(secret, &p, &w)
        ));
        assert!(worker_accepts(
            secret,
            &p,
            &w,
            &parent_proof(secret, &p, &w)
        ));
    }

    #[test]
    fn a_wrong_or_missing_secret_is_refused() {
        let (p, w) = (nonce(), nonce());
        let mine = Some("right");
        assert!(!parent_accepts(
            mine,
            &p,
            &w,
            &worker_proof(Some("wrong"), &p, &w)
        ));
        assert!(!parent_accepts(mine, &p, &w, &worker_proof(None, &p, &w)));
        assert!(!worker_accepts(
            mine,
            &p,
            &w,
            &parent_proof(Some("wrong"), &p, &w)
        ));
        assert!(!worker_accepts(mine, &p, &w, &parent_proof(None, &p, &w)));
    }

    /// One side's proof cannot stand in for the other's, nor for another
    /// connection's.
    #[test]
    fn a_proof_cannot_be_replayed() {
        let (p, w) = (nonce(), nonce());
        let secret = Some("s3cret");
        assert!(!worker_accepts(
            secret,
            &p,
            &w,
            &worker_proof(secret, &p, &w)
        ));
        assert!(!parent_accepts(
            secret,
            &p,
            &w,
            &parent_proof(secret, &p, &w)
        ));
        let other = nonce();
        assert!(!parent_accepts(
            secret,
            &other,
            &w,
            &worker_proof(secret, &p, &w)
        ));
    }

    #[test]
    fn without_a_secret_anything_is_accepted() {
        let (p, w) = (nonce(), nonce());
        assert!(parent_accepts(None, &p, &w, &[7; 32]));
        assert!(worker_accepts(None, &p, &w, &[0; 32]));
        assert_ne!(nonce(), nonce());
    }
}
