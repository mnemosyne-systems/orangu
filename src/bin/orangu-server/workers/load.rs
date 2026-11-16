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

//! How busy a node is, from `0.0` (idle) to `1.0` (fully busy): the
//! higher of its slots in use and its processors' busy time, so a full
//! slot table and a machine busy with something else both count.

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

/// How often the processors are sampled.
#[cfg_attr(test, allow(dead_code))]
pub const SAMPLE_EVERY: Duration = Duration::from_secs(1);

/// The processors' busy share, as the last sample found it.
static BUSY: AtomicU32 = AtomicU32::new(0);

/// A node's load from its slots in use and its processors' busy share.
pub fn combine(in_use: usize, slots: usize, busy: f32) -> f32 {
    let slots_share = if slots == 0 {
        0.0
    } else {
        in_use as f32 / slots as f32
    };
    slots_share.max(busy).clamp(0.0, 1.0)
}

/// The processors' busy share as the sampler last saw it.
pub fn busy() -> f32 {
    f32::from_bits(BUSY.load(Ordering::Relaxed))
}

/// Samples the processors every [`SAMPLE_EVERY`] on a thread of its own,
/// for as long as the process runs.
#[cfg_attr(test, allow(dead_code))]
pub fn start_sampler() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("orangu-workers-load".to_string())
            .spawn(|| {
                // Without a view of the processors, slots in use alone say
                // how busy the node is.
                let Ok(mut system) = sysinfo::System::new() else {
                    return;
                };
                loop {
                    system.refresh_cpu_usage();
                    std::thread::sleep(SAMPLE_EVERY.max(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL));
                    system.refresh_cpu_usage();
                    let cpu = system.global_cpu_usage() / 100.0;
                    let busy = cpu.max(gpu_busy(Path::new("/sys/class/drm")).unwrap_or(0.0));
                    BUSY.store(busy.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
                }
            });
    });
}

/// The busiest GPU's share of busy time, where the driver reports one
/// (`gpu_busy_percent` under each card), `None` where none does.
fn gpu_busy(drm: &Path) -> Option<f32> {
    std::fs::read_dir(drm)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|card| {
            std::fs::read_to_string(card.path().join("device/gpu_busy_percent")).ok()
        })
        .filter_map(|text| text.trim().parse::<f32>().ok())
        .map(|percent| percent / 100.0)
        .reduce(f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_load_is_the_busier_of_slots_and_processors() {
        assert_eq!(combine(0, 4, 0.0), 0.0);
        assert_eq!(combine(1, 4, 0.1), 0.25);
        assert_eq!(combine(1, 4, 0.9), 0.9);
        assert_eq!(combine(6, 4, 0.0), 1.0);
        assert_eq!(combine(1, 0, 0.3), 0.3);
    }

    #[test]
    fn the_busiest_card_counts() {
        let dir = tempfile::tempdir().unwrap();
        for (card, percent) in [("card0", "12\n"), ("card1", "80\n")] {
            let device = dir.path().join(card).join("device");
            std::fs::create_dir_all(&device).unwrap();
            std::fs::write(device.join("gpu_busy_percent"), percent).unwrap();
        }
        std::fs::create_dir_all(dir.path().join("renderD128")).unwrap();
        assert_eq!(gpu_busy(dir.path()), Some(0.8));
        assert_eq!(gpu_busy(&dir.path().join("none")), None);
    }
}
