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

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use serde::Serialize;
use std::sync::Arc;

use super::WebState;

pub fn router() -> Router<Arc<WebState>> {
    Router::new().route("/api/monitor", get(monitor))
}

#[derive(Serialize)]
struct OsView {
    name: String,
    version: Option<String>,
    kernel: Option<String>,
    hostname: Option<String>,
    uptime_seconds: u64,
    load_one: Option<f64>,
    load_five: Option<f64>,
    load_fifteen: Option<f64>,
}

#[derive(Serialize)]
struct CpuView {
    brand: String,
    vendor: String,
    arch: String,
    physical_cores: Option<usize>,
    logical_cores: usize,
    frequency_mhz: u64,
    governor: Option<String>,
    /// Host-wide CPU usage, 0–100, sampled over [`SAMPLE_MS`] inside the
    /// blocking task (`sysinfo` needs two refreshes to report a rate).
    usage_pct: f32,
    per_core_pct: Vec<f32>,
}

#[derive(Serialize)]
struct MemoryView {
    total_bytes: u64,
    available_bytes: u64,
    /// This server process's own RSS, when the platform reports one.
    process_bytes: Option<u64>,
    swap_total_bytes: u64,
    swap_used_bytes: u64,
}

#[derive(Serialize)]
struct GpuView {
    vendor: String,
    name: String,
    total_bytes: Option<u64>,
    used_bytes: Option<u64>,
    driver: Option<String>,
    memory_kind: &'static str,
}

#[derive(Serialize)]
struct NpuView {
    vendor: String,
    target: String,
    cores: u32,
    driver: Option<String>,
    stack: &'static str,
}

#[derive(Serialize)]
struct PowerView {
    source: &'static str,
    battery_pct: Option<u8>,
    thermals: Vec<ThermalView>,
}

#[derive(Serialize)]
struct ThermalView {
    label: String,
    celsius: f32,
    critical_celsius: Option<f32>,
}

#[derive(Serialize)]
struct ServerView {
    model: String,
    architecture: String,
    backend: String,
    role: String,
    slots_total: usize,
    slots_busy: usize,
    queued: usize,
    queue_limit: usize,
}

/// Sampling window for the live CPU rate, in milliseconds. Long enough for
/// `sysinfo` to report a real rate, short enough that a 2 s UI poll never
/// feels laggy because of it.
const SAMPLE_MS: u64 = 250;

async fn monitor(State(state): State<Arc<WebState>>) -> impl IntoResponse {
    let server = ServerView {
        model: state.model_display.clone(),
        architecture: state.architecture.clone(),
        backend: state.backend_label.clone(),
        role: state.engine.role.label().to_string(),
        slots_total: state.engine.slots.total(),
        slots_busy: state.engine.slots.busy_count(),
        queued: state.engine.slots.queued(),
        queue_limit: state.engine.slots.queue_limit(),
    };
    let snapshot = tokio::task::spawn_blocking(collect).await;
    match snapshot {
        Ok(Ok(mut view)) => {
            view.server = Some(server);
            Json(view).into_response()
        }
        Ok(Err(err)) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

#[derive(Serialize)]
struct SnapshotView {
    os: OsView,
    cpu: CpuView,
    memory: MemoryView,
    gpus: Vec<GpuView>,
    npu: Option<NpuView>,
    power: PowerView,
    #[serde(skip_serializing_if = "Option::is_none")]
    server: Option<ServerView>,
}

fn collect() -> anyhow::Result<SnapshotView> {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

    let os = orangu::os::detect();
    let cpu = orangu::hardware::detect_cpu();
    let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
    let npu = orangu::npu::detect_npu_inventory();
    let power = orangu::hardware::detect_power();

    // Two-refresh sampling: the first read arms the counters, the sleep
    // lets them accumulate, the second reports the rate over the window.
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything()),
    );
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    std::thread::sleep(std::time::Duration::from_millis(SAMPLE_MS));
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);

    let usage_pct = sys.global_cpu_usage();
    let per_core_pct = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
    let pid = sysinfo::get_current_pid().ok();
    let process_bytes = pid.and_then(|pid| sys.process(pid)).map(|p| p.memory());

    Ok(SnapshotView {
        os: OsView {
            name: os.name.clone(),
            version: os.version.clone(),
            kernel: os.kernel.clone(),
            hostname: os.hostname.clone(),
            uptime_seconds: os.uptime_seconds,
            load_one: os.load_average.as_ref().map(|l| l.one),
            load_five: os.load_average.as_ref().map(|l| l.five),
            load_fifteen: os.load_average.as_ref().map(|l| l.fifteen),
        },
        cpu: CpuView {
            brand: cpu.brand.clone(),
            vendor: cpu.vendor.clone(),
            arch: cpu.arch.clone(),
            physical_cores: cpu.physical_cores,
            logical_cores: cpu.logical_cores,
            frequency_mhz: cpu.frequency_mhz,
            governor: orangu::hardware::cpu_governor(),
            usage_pct,
            per_core_pct,
        },
        memory: MemoryView {
            total_bytes: sys.total_memory(),
            available_bytes: sys.available_memory(),
            process_bytes,
            swap_total_bytes: sys.total_swap(),
            swap_used_bytes: sys.used_swap(),
        },
        gpus: gpus
            .iter()
            .map(|g| GpuView {
                vendor: g.vendor.clone(),
                name: g.name.clone(),
                total_bytes: g.vram_total_bytes,
                used_bytes: g.vram_used_bytes,
                driver: g.driver.clone(),
                memory_kind: match g.memory_kind {
                    orangu::hardware::MemoryKind::Dedicated => "dedicated",
                    orangu::hardware::MemoryKind::Shared => "shared",
                    orangu::hardware::MemoryKind::Unknown => "unknown",
                },
            })
            .collect(),
        npu: npu.as_ref().map(|n| NpuView {
            vendor: n.vendor.clone(),
            target: n.target.clone(),
            cores: n.cores,
            driver: n.driver.clone(),
            stack: match n.stack {
                orangu::npu::NpuStack::Noe => "noe",
                orangu::npu::NpuStack::Rknpu => "rknpu",
            },
        }),
        power: PowerView {
            source: match power.source {
                orangu::hardware::PowerSource::Mains => "mains",
                orangu::hardware::PowerSource::Battery => "battery",
                orangu::hardware::PowerSource::Unknown => "unknown",
            },
            battery_pct: power.battery_percent,
            thermals: power
                .thermals
                .iter()
                .take(8)
                .map(|t| ThermalView {
                    label: t.label.clone(),
                    celsius: t.celsius,
                    critical_celsius: t.critical_celsius,
                })
                .collect(),
        },
        server: None,
    })
}
