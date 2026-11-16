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

//! Finding the other nodes of a workers channel (`[workers].name`).
//!
//! A node scans every address of its local IPv4 networks on its own
//! `[workers].port` and sends each listener a [`Message::Probe`] carrying a
//! hash of its channel. A node on the same channel answers with a
//! [`Message::ProbeAck`]: its id, its name and whether it is free to be
//! claimed. Any other listener closes the connection. The channel name
//! never crosses the network, and being found grants nothing: claiming a
//! node is the ordinary handshake, with `[workers].secret`.

use super::protocol::{Message, ProbeState, read_message, write_message};
use super::transport::{self, Tls};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// The widest network a node scans: a wider one is scanned only in the
/// /22 around the node's own address.
pub const WIDEST_PREFIX: u8 = 22;

/// How long dialing one address may take.
pub const CONNECT_TIMEOUT: Duration = Duration::from_millis(300);

/// How long a dialed node may take to answer a probe.
pub const ANSWER_TIMEOUT: Duration = Duration::from_secs(2);

/// Addresses dialed at once.
pub const IN_FLIGHT: usize = 64;

/// What a node sends and compares instead of its channel's name.
pub fn channel_hash(name: &str) -> [u8; 32] {
    super::auth::hmac_sha256(b"orangu-workers-channel", name.as_bytes())
}

/// A node on the channel, as its answer to a probe described it.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    /// Where it was reached, `ip:port`.
    pub addr: String,
    /// Its id for this run of its process.
    pub node: String,
    /// Its `[workers]` name.
    pub name: String,
    pub state: ProbeState,
    /// How busy it is, `0.0` to `1.0`.
    pub load: f32,
    /// Whether it can serve the model alone.
    pub whole: bool,
    /// Where its API answers: a request routed to it goes there.
    pub api: String,
    /// The model it serves, by its layout, and by samples of its weights
    /// (zeros when it has only part of the model).
    pub model: [u8; 32],
    pub content: [u8; 32],
}

impl Found {
    /// Its API's base URL, `http://ip:port` or `https://ip:port`.
    fn api_url(ip: std::net::IpAddr, port: u16, tls: bool) -> String {
        let scheme = if tls { "https" } else { "http" };
        match ip {
            std::net::IpAddr::V6(v6) => format!("{scheme}://[{v6}]:{port}"),
            std::net::IpAddr::V4(v4) => format!("{scheme}://{v4}:{port}"),
        }
    }
}

/// The addresses of the network `ip`/`prefix` other than `ip` itself, its
/// network address and its broadcast address — within the /22 around `ip`
/// when the network is wider.
pub fn hosts(ip: Ipv4Addr, prefix: u8) -> Vec<Ipv4Addr> {
    let own = u32::from(ip);
    network_hosts(ip, prefix.clamp(WIDEST_PREFIX, 32))
        .into_iter()
        .filter(|&host| u32::from(host) != own)
        .collect()
}

/// The widest network `[workers].network` may name.
pub const WIDEST_NAMED_PREFIX: u8 = 16;

/// Every host address of the network `ip`/`prefix`: all but its network
/// and broadcast addresses, both ends of a /31, nothing for a /32.
pub fn network_hosts(ip: Ipv4Addr, prefix: u8) -> Vec<Ipv4Addr> {
    let prefix = prefix.min(32);
    if prefix == 32 {
        return Vec::new();
    }
    let mask = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - prefix)
    };
    let network = u32::from(ip) & mask;
    let broadcast = network | !mask;
    // A /31 has two hosts and neither network nor broadcast address.
    let (first, last) = if prefix == 31 {
        (network, broadcast)
    } else {
        (network + 1, broadcast - 1)
    };
    (first..=last).map(Ipv4Addr::from).collect()
}

/// Parses `[workers].network`: comma-separated IPv4 networks, each
/// `address/prefix`, none wider than [`WIDEST_NAMED_PREFIX`].
pub fn parse_networks(value: &str) -> anyhow::Result<Vec<(Ipv4Addr, u8)>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            let (ip, prefix) = entry
                .split_once('/')
                .ok_or_else(|| anyhow::anyhow!("'{entry}' is not address/prefix"))?;
            let ip: Ipv4Addr = ip
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("'{entry}' does not start with an IPv4 address"))?;
            let prefix: u8 = prefix
                .trim()
                .parse()
                .ok()
                .filter(|p| *p <= 32)
                .ok_or_else(|| anyhow::anyhow!("'{entry}' has no prefix of 0-32"))?;
            anyhow::ensure!(
                prefix >= WIDEST_NAMED_PREFIX,
                "'{entry}' is wider than /{WIDEST_NAMED_PREFIX}"
            );
            Ok((ip, prefix))
        })
        .collect()
}

/// The addresses a scan looks at: every host of `networks` when there
/// are any, this machine's own networks ([`local_hosts`]) otherwise.
pub fn scan_hosts(networks: &[(Ipv4Addr, u8)]) -> Vec<Ipv4Addr> {
    if networks.is_empty() {
        return local_hosts();
    }
    let mut out: Vec<Ipv4Addr> = networks
        .iter()
        .flat_map(|&(ip, prefix)| network_hosts(ip, prefix))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Every address of this machine's IPv4 networks, loopback aside, each
/// once.
pub fn local_hosts() -> Vec<Ipv4Addr> {
    let mut out: Vec<Ipv4Addr> = Vec::new();
    for interface in if_addrs::get_if_addrs().unwrap_or_default() {
        let if_addrs::IfAddr::V4(v4) = interface.addr else {
            continue;
        };
        if v4.ip.is_loopback() || v4.ip.is_link_local() {
            continue;
        }
        out.extend(hosts(v4.ip, v4.prefixlen));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Asks the listener at `addr` whether it is on `channel`. `None` when
/// nothing listens there, it is on another channel, or it does not
/// understand the question.
pub fn probe(addr: SocketAddr, channel: &[u8; 32], tls: Option<&Tls>) -> Option<Found> {
    let tcp = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).ok()?;
    tcp.set_read_timeout(Some(ANSWER_TIMEOUT)).ok()?;
    tcp.set_write_timeout(Some(ANSWER_TIMEOUT)).ok()?;
    let _ = tcp.set_nodelay(true);
    let shown = addr.to_string();
    let split = transport::client(tls, tcp, &shown).ok()?;
    let mut writer = split.writer;
    let mut reader = split.reader;
    write_message(&mut *writer, 0, &Message::Probe { channel: *channel }).ok()?;
    let answer = read_message(&mut *reader).ok();
    let _ = split.tcp.shutdown(std::net::Shutdown::Both);
    match answer? {
        Message::ProbeAck {
            node,
            name,
            state,
            load,
            whole,
            api_port,
            api_tls,
            model,
            content,
        } => Some(Found {
            addr: shown,
            node,
            name,
            state,
            load: if load.is_finite() {
                load.clamp(0.0, 1.0)
            } else {
                1.0
            },
            whole,
            api: Found::api_url(addr.ip(), api_port, api_tls),
            model,
            content,
        }),
        _ => None,
    }
}

/// Probes `port` on every one of `hosts`, [`IN_FLIGHT`] at a time, and
/// returns the nodes on `channel` ordered by name, then address: an order
/// that does not change when a node restarts.
pub fn scan(hosts: &[Ipv4Addr], port: u16, channel: &[u8; 32], tls: Option<&Tls>) -> Vec<Found> {
    let next = AtomicUsize::new(0);
    let found = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..IN_FLIGHT.min(hosts.len()) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&ip) = hosts.get(i) else { break };
                    let addr = SocketAddr::V4(SocketAddrV4::new(ip, port));
                    if let Some(node) = probe(addr, channel, tls) {
                        found.lock().unwrap().push(node);
                    }
                }
            });
        }
    });
    let mut found = found.into_inner().unwrap();
    found.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.addr.cmp(&b.addr)));
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;
    use std::net::TcpListener;

    #[test]
    fn a_24_is_its_254_hosts_but_this_one() {
        let ip = Ipv4Addr::new(192, 168, 1, 10);
        let all = hosts(ip, 24);
        assert_eq!(all.len(), 253);
        assert_eq!(all[0], Ipv4Addr::new(192, 168, 1, 1));
        assert_eq!(*all.last().unwrap(), Ipv4Addr::new(192, 168, 1, 254));
        assert!(!all.contains(&ip));
    }

    #[test]
    fn a_wide_network_is_scanned_in_the_22_around_this_node() {
        let all = hosts(Ipv4Addr::new(10, 1, 6, 7), 8);
        assert_eq!(all.len(), 1021);
        assert_eq!(all[0], Ipv4Addr::new(10, 1, 4, 1));
        assert_eq!(*all.last().unwrap(), Ipv4Addr::new(10, 1, 7, 254));
    }

    #[test]
    fn point_to_point_links_have_the_other_end_only() {
        assert_eq!(
            hosts(Ipv4Addr::new(10, 0, 0, 0), 31),
            vec![Ipv4Addr::new(10, 0, 0, 1)]
        );
        assert!(hosts(Ipv4Addr::new(10, 0, 0, 1), 32).is_empty());
    }

    #[test]
    fn a_named_network_is_scanned_whole() {
        let hosts = scan_hosts(&parse_networks("10.0.0.0/22, 10.0.1.0/24").unwrap());
        assert_eq!(hosts.len(), 1022);
        assert_eq!(hosts[0], Ipv4Addr::new(10, 0, 0, 1));
        assert_eq!(*hosts.last().unwrap(), Ipv4Addr::new(10, 0, 3, 254));
        assert_eq!(network_hosts(Ipv4Addr::new(10, 0, 0, 0), 16).len(), 65534);
    }

    #[test]
    fn a_bad_network_is_refused() {
        for bad in [
            "10.0.0.0",
            "10.0.0.0/33",
            "fd00::/64",
            "host/24",
            "10.0.0.0/8",
        ] {
            assert!(parse_networks(bad).is_err(), "{bad}");
        }
        assert!(parse_networks(" ").unwrap().is_empty());
    }

    #[test]
    fn channels_hash_apart() {
        assert_ne!(channel_hash("lab"), channel_hash("lab2"));
        assert_eq!(channel_hash("lab"), channel_hash("lab"));
    }

    /// A listener that answers one probe for `channel` as `name`, and
    /// closes on any other.
    fn answering(channel: &'static str, name: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let mut writer = stream.try_clone().unwrap();
                let mut reader = BufReader::new(stream);
                if let Ok(Message::Probe { channel: asked }) = read_message(&mut reader)
                    && asked == channel_hash(channel)
                {
                    let _ = write_message(
                        &mut writer,
                        0,
                        &Message::ProbeAck {
                            node: "n1".to_string(),
                            name: name.to_string(),
                            state: ProbeState::Idle,
                            load: 0.5,
                            whole: true,
                            api_port: 8100,
                            api_tls: false,
                            model: [1; 32],
                            content: [2; 32],
                        },
                    );
                }
            }
        });
        port
    }

    #[test]
    fn a_node_on_the_channel_is_found_and_one_on_another_is_not() {
        let port = answering("lab", "beta:8400");
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let found = probe(addr, &channel_hash("lab"), None).expect("on the channel");
        assert_eq!(found.name, "beta:8400");
        assert_eq!(found.state, ProbeState::Idle);
        assert_eq!(found.load, 0.5);
        assert!(found.whole);
        assert_eq!(found.api, "http://127.0.0.1:8100");
        assert!(probe(addr, &channel_hash("other"), None).is_none());
    }

    #[test]
    fn nothing_listening_is_nothing_found() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let found = scan(&[Ipv4Addr::LOCALHOST], port, &channel_hash("lab"), None);
        assert!(found.is_empty());
    }

    #[test]
    fn a_scan_finds_the_listener() {
        let port = answering("lab", "gamma:8400");
        let found = scan(&[Ipv4Addr::LOCALHOST], port, &channel_hash("lab"), None);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].addr, format!("127.0.0.1:{port}"));
    }
}
