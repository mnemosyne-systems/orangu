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

//! A generated clip as a file a browser plays: the frames encoded as
//! **AV1** by `rav1e` (pure Rust, no system codec), the sound as **FLAC**
//! ([`super::flac`]), both in one **MP4** whose index (`moov`) comes before
//! the media (`mdat`), so a `<video>` element starts playing before the
//! whole file has arrived.
//!
//! Frames are converted to 4:2:0 BT.709 limited-range YUV and the stream is
//! tagged so (in the AV1 sequence header and in a `colr` box). Every sample
//! of the video track is one AV1 temporal unit with its temporal delimiter
//! removed, as AV1-in-ISOBMFF asks.

use super::flac::Flac;
use anyhow::{Context, Result, anyhow, bail, ensure};

/// Encoder knobs a request may set.
#[derive(Debug, Clone, Copy)]
pub struct Av1Settings {
    /// `rav1e`'s base quantizer, 0 (best) to 255.
    pub quantizer: usize,
    /// `rav1e`'s speed preset, 0 (slowest) to 10.
    pub speed: u8,
    /// Encoder threads; 0 is rayon's global pool.
    pub threads: usize,
}

impl Default for Av1Settings {
    fn default() -> Self {
        Self {
            quantizer: 80,
            speed: 10,
            threads: 0,
        }
    }
}

/// The encoded video track.
pub struct Av1Track {
    /// `AV1CodecConfigurationRecord` (the body of `av1C`).
    pub config: Vec<u8>,
    /// Each frame's temporal unit and whether it is a key frame.
    pub samples: Vec<(Vec<u8>, bool)>,
    pub width: u32,
    pub height: u32,
}

/// RGB `[0, 1]` (`[height][width][3]`) to 8-bit 4:2:0 BT.709 limited range.
fn rgb_to_yuv420(rgb: &[f32], width: usize, height: usize) -> [Vec<u8>; 3] {
    let mut y = vec![0u8; width * height];
    let (cw, ch) = (width / 2, height / 2);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    let px = |i: usize| -> (f32, f32, f32) { (rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]) };
    for (i, out) in y.iter_mut().enumerate() {
        let (r, g, b) = px(i);
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        *out = (16.0 + 219.0 * luma).round().clamp(0.0, 255.0) as u8;
    }
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let (pr, pg, pb) = px((cy * 2 + dy) * width + cx * 2 + dx);
                r += pr;
                g += pg;
                b += pb;
            }
            let (r, g, b) = (r / 4.0, g / 4.0, b / 4.0);
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let cb = (b - luma) / 1.8556;
            let cr = (r - luma) / 1.5748;
            u[cy * cw + cx] = (128.0 + 224.0 * cb).round().clamp(0.0, 255.0) as u8;
            v[cy * cw + cx] = (128.0 + 224.0 * cr).round().clamp(0.0, 255.0) as u8;
        }
    }
    [y, u, v]
}

/// Drops temporal-delimiter OBUs from a temporal unit.
fn strip_temporal_delimiters(tu: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(tu.len());
    let mut at = 0;
    while at < tu.len() {
        let start = at;
        let header = tu[at];
        let obu_type = (header >> 3) & 0x0F;
        let extension = header & 0x04 != 0;
        let has_size = header & 0x02 != 0;
        at += 1 + usize::from(extension);
        ensure!(has_size, "an AV1 OBU without a size field");
        let mut size = 0u64;
        for i in 0..8 {
            let byte = *tu.get(at).ok_or_else(|| anyhow!("a truncated AV1 OBU"))?;
            at += 1;
            size |= u64::from(byte & 0x7F) << (7 * i);
            if byte & 0x80 == 0 {
                break;
            }
        }
        let end = at + size as usize;
        ensure!(end <= tu.len(), "an AV1 OBU runs past its temporal unit");
        if obu_type != 2 {
            out.extend_from_slice(&tu[start..end]);
        }
        at = end;
    }
    Ok(out)
}

/// Encodes `frames` (each RGB `[0, 1]`, `[height][width][3]`) at `fps`.
pub fn encode_av1(
    frames: &[Vec<f32>],
    width: usize,
    height: usize,
    fps: u32,
    settings: Av1Settings,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Av1Track> {
    use rav1e::color::{
        ChromaSampling, ColorDescription, ColorPrimaries, MatrixCoefficients, PixelRange,
        TransferCharacteristics,
    };
    use rav1e::config::{Config, EncoderConfig};
    use rav1e::data::Rational;
    ensure!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "a 4:2:0 frame needs even sides"
    );
    let mut enc = EncoderConfig::with_speed_preset(settings.speed.min(10));
    enc.width = width;
    enc.height = height;
    enc.bit_depth = 8;
    enc.chroma_sampling = ChromaSampling::Cs420;
    enc.time_base = Rational::new(1, u64::from(fps.max(1)));
    enc.pixel_range = PixelRange::Limited;
    enc.color_description = Some(ColorDescription {
        color_primaries: ColorPrimaries::BT709,
        transfer_characteristics: TransferCharacteristics::BT709,
        matrix_coefficients: MatrixCoefficients::BT709,
    });
    enc.quantizer = settings.quantizer.min(255);
    enc.min_key_frame_interval = 12;
    enc.max_key_frame_interval = 240;
    enc.low_latency = false;
    let config = Config::new()
        .with_encoder_config(enc)
        .with_threads(settings.threads);
    let mut ctx: rav1e::Context<u8> = config
        .new_context()
        .map_err(|e| anyhow!("configuring the AV1 encoder: {e}"))?;
    let mut samples = Vec::with_capacity(frames.len());
    let receive =
        |ctx: &mut rav1e::Context<u8>, samples: &mut Vec<(Vec<u8>, bool)>| -> Result<bool> {
            loop {
                match ctx.receive_packet() {
                    Ok(packet) => {
                        let key = packet.frame_type == rav1e::data::FrameType::KEY;
                        samples.push((strip_temporal_delimiters(&packet.data)?, key));
                    }
                    Err(rav1e::data::EncoderStatus::Encoded) => continue,
                    Err(rav1e::data::EncoderStatus::NeedMoreData) => return Ok(false),
                    Err(rav1e::data::EncoderStatus::LimitReached) => return Ok(true),
                    Err(e) => bail!("the AV1 encoder failed: {e}"),
                }
            }
        };
    for rgb in frames {
        if super::cancelled(cancel) {
            bail!("cancelled");
        }
        ensure!(
            rgb.len() == width * height * 3,
            "a frame is not {width} × {height} RGB"
        );
        let planes = rgb_to_yuv420(rgb, width, height);
        let mut frame = ctx.new_frame();
        let widths = [width, width / 2, width / 2];
        for ((plane, data), w) in frame.planes.iter_mut().zip(&planes).zip(widths) {
            plane.copy_from_raw_u8(data, w, 1);
        }
        ctx.send_frame(frame)
            .map_err(|e| anyhow!("the AV1 encoder refused a frame: {e}"))?;
        receive(&mut ctx, &mut samples)?;
    }
    ctx.flush();
    while !receive(&mut ctx, &mut samples)? {}
    ensure!(
        samples.len() == frames.len(),
        "the AV1 encoder returned {} frames for {}",
        samples.len(),
        frames.len()
    );
    Ok(Av1Track {
        config: ctx.container_sequence_header(),
        samples,
        width: width as u32,
        height: height as u32,
    })
}

// ---------------------------------------------------------------------------
// ISO base media file format
// ---------------------------------------------------------------------------

fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + body.len());
    out.extend(((8 + body.len()) as u32).to_be_bytes());
    out.extend(kind);
    out.extend(body);
    out
}

fn full_box(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(4 + body.len());
    b.push(version);
    b.extend(&flags.to_be_bytes()[1..]);
    b.extend(body);
    boxed(kind, &b)
}

fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.concat()
}

const MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

fn mvhd(timescale: u32, duration: u32, next_track: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0u32.to_be_bytes()); // creation
    b.extend(0u32.to_be_bytes()); // modification
    b.extend(timescale.to_be_bytes());
    b.extend(duration.to_be_bytes());
    b.extend(0x0001_0000u32.to_be_bytes()); // rate 1.0
    b.extend(0x0100u16.to_be_bytes()); // volume 1.0
    b.extend([0u8; 10]);
    for m in MATRIX {
        b.extend(m.to_be_bytes());
    }
    b.extend([0u8; 24]);
    b.extend(next_track.to_be_bytes());
    full_box(b"mvhd", 0, 0, &b)
}

fn tkhd(id: u32, duration: u32, audio: bool, width: u32, height: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0u32.to_be_bytes());
    b.extend(0u32.to_be_bytes());
    b.extend(id.to_be_bytes());
    b.extend(0u32.to_be_bytes());
    b.extend(duration.to_be_bytes());
    b.extend([0u8; 8]);
    b.extend(0u16.to_be_bytes()); // layer
    b.extend(0u16.to_be_bytes()); // alternate group
    b.extend(if audio { 0x0100u16 } else { 0 }.to_be_bytes());
    b.extend(0u16.to_be_bytes());
    for m in MATRIX {
        b.extend(m.to_be_bytes());
    }
    b.extend((width << 16).to_be_bytes());
    b.extend((height << 16).to_be_bytes());
    full_box(b"tkhd", 0, 0x7, &b)
}

fn mdhd(timescale: u32, duration: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0u32.to_be_bytes());
    b.extend(0u32.to_be_bytes());
    b.extend(timescale.to_be_bytes());
    b.extend(duration.to_be_bytes());
    b.extend(0x55C4u16.to_be_bytes()); // language "und"
    b.extend(0u16.to_be_bytes());
    full_box(b"mdhd", 0, 0, &b)
}

fn hdlr(kind: &[u8; 4], name: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0u32.to_be_bytes());
    b.extend(kind);
    b.extend([0u8; 12]);
    b.extend(name.as_bytes());
    b.push(0);
    full_box(b"hdlr", 0, 0, &b)
}

fn dinf() -> Vec<u8> {
    let url = full_box(b"url ", 0, 1, &[]);
    let mut dref = 1u32.to_be_bytes().to_vec();
    dref.extend(url);
    boxed(b"dinf", &full_box(b"dref", 0, 0, &dref))
}

/// `stts` from per-sample durations, run-length coded.
fn stts(durations: &[u32]) -> Vec<u8> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &d in durations {
        match runs.last_mut() {
            Some((count, delta)) if *delta == d => *count += 1,
            _ => runs.push((1, d)),
        }
    }
    let mut b = (runs.len() as u32).to_be_bytes().to_vec();
    for (count, delta) in runs {
        b.extend(count.to_be_bytes());
        b.extend(delta.to_be_bytes());
    }
    full_box(b"stts", 0, 0, &b)
}

/// The sample table for samples stored contiguously from `offset`, one
/// chunk.
fn sample_tables(sizes: &[u32], durations: &[u32], offset: u64, keys: Option<&[u32]>) -> Vec<u8> {
    let mut stsc = 1u32.to_be_bytes().to_vec();
    stsc.extend(1u32.to_be_bytes());
    stsc.extend((sizes.len() as u32).to_be_bytes());
    stsc.extend(1u32.to_be_bytes());
    let mut stsz = 0u32.to_be_bytes().to_vec();
    stsz.extend((sizes.len() as u32).to_be_bytes());
    for s in sizes {
        stsz.extend(s.to_be_bytes());
    }
    let mut co64 = 1u32.to_be_bytes().to_vec();
    co64.extend(offset.to_be_bytes());
    let mut parts = vec![
        stts(durations),
        full_box(b"stsc", 0, 0, &stsc),
        full_box(b"stsz", 0, 0, &stsz),
        full_box(b"co64", 0, 0, &co64),
    ];
    if let Some(keys) = keys {
        let mut stss = (keys.len() as u32).to_be_bytes().to_vec();
        for k in keys {
            stss.extend(k.to_be_bytes());
        }
        parts.push(full_box(b"stss", 0, 0, &stss));
    }
    cat(&parts)
}

fn av01_entry(track: &Av1Track) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend([0u8; 6]);
    b.extend(1u16.to_be_bytes()); // data reference index
    b.extend([0u8; 16]);
    b.extend((track.width as u16).to_be_bytes());
    b.extend((track.height as u16).to_be_bytes());
    b.extend(0x0048_0000u32.to_be_bytes()); // 72 dpi
    b.extend(0x0048_0000u32.to_be_bytes());
    b.extend(0u32.to_be_bytes());
    b.extend(1u16.to_be_bytes()); // frame count
    let mut name = [0u8; 32];
    let label = b"orangu AV1";
    name[0] = label.len() as u8;
    name[1..=label.len()].copy_from_slice(label);
    b.extend(name);
    b.extend(0x0018u16.to_be_bytes()); // depth
    b.extend((-1i16).to_be_bytes());
    b.extend(boxed(b"av1C", &track.config));
    // nclx: BT.709 primaries, transfer and matrix, limited range.
    let mut colr = b"nclx".to_vec();
    colr.extend(1u16.to_be_bytes());
    colr.extend(1u16.to_be_bytes());
    colr.extend(1u16.to_be_bytes());
    colr.push(0);
    b.extend(boxed(b"colr", &colr));
    boxed(b"av01", &b)
}

fn flac_entry(flac: &Flac) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend([0u8; 6]);
    b.extend(1u16.to_be_bytes());
    b.extend([0u8; 8]);
    b.extend((flac.channels as u16).to_be_bytes());
    b.extend(16u16.to_be_bytes());
    b.extend(0u16.to_be_bytes());
    b.extend(0u16.to_be_bytes());
    b.extend((flac.sample_rate.min(65535) << 16).to_be_bytes());
    // dfLa: one STREAMINFO block, marked last.
    let mut dfla = vec![0x80u8];
    dfla.extend(&(flac.streaminfo.len() as u32).to_be_bytes()[1..]);
    dfla.extend(&flac.streaminfo);
    b.extend(full_box(b"dfLa", 0, 0, &dfla));
    boxed(b"fLaC", &b)
}

/// The MP4 file: the video track and, when there is one, the sound.
pub fn write(video: &Av1Track, fps: u32, audio: Option<&Flac>) -> Result<Vec<u8>> {
    ensure!(!video.samples.is_empty(), "a clip with no frames");
    let video_scale = fps.max(1) * 1000;
    let frame = 1000u32;
    let video_duration = frame * video.samples.len() as u32;
    let video_sizes: Vec<u32> = video.samples.iter().map(|s| s.0.len() as u32).collect();
    let keys: Vec<u32> = video
        .samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.1)
        .map(|(i, _)| i as u32 + 1)
        .collect();
    let video_bytes: u64 = video_sizes.iter().map(|&s| s as u64).sum();
    let audio_sizes: Vec<u32> = audio
        .map(|a| a.frames.iter().map(|f| f.0.len() as u32).collect())
        .unwrap_or_default();
    let audio_durations: Vec<u32> = audio
        .map(|a| a.frames.iter().map(|f| f.1 as u32).collect())
        .unwrap_or_default();
    let audio_total: u64 = audio_durations.iter().map(|&d| d as u64).sum();

    let ftyp = {
        let mut b = b"isom".to_vec();
        b.extend(0x200u32.to_be_bytes());
        for brand in [b"isom", b"iso6", b"av01", b"mp41"] {
            b.extend(brand);
        }
        boxed(b"ftyp", &b)
    };
    let movie_scale = 1000u32;
    let movie_duration =
        (u64::from(video_duration) * u64::from(movie_scale) / u64::from(video_scale)) as u32;
    let build_moov = |video_offset: u64, audio_offset: u64| -> Vec<u8> {
        let vmhd = full_box(b"vmhd", 0, 1, &[0u8; 8]);
        let video_stbl = boxed(
            b"stbl",
            &cat(&[
                full_box(
                    b"stsd",
                    0,
                    0,
                    &cat(&[1u32.to_be_bytes().to_vec(), av01_entry(video)]),
                ),
                sample_tables(
                    &video_sizes,
                    &vec![frame; video_sizes.len()],
                    video_offset,
                    Some(&keys),
                ),
            ]),
        );
        let video_trak = boxed(
            b"trak",
            &cat(&[
                tkhd(1, movie_duration, false, video.width, video.height),
                boxed(
                    b"mdia",
                    &cat(&[
                        mdhd(video_scale, video_duration),
                        hdlr(b"vide", "orangu video"),
                        boxed(b"minf", &cat(&[vmhd, dinf(), video_stbl])),
                    ]),
                ),
            ]),
        );
        let mut parts = vec![
            mvhd(
                movie_scale,
                movie_duration,
                if audio.is_some() { 3 } else { 2 },
            ),
            video_trak,
        ];
        if let Some(a) = audio {
            let smhd = full_box(b"smhd", 0, 0, &[0u8; 4]);
            let audio_stbl = boxed(
                b"stbl",
                &cat(&[
                    full_box(
                        b"stsd",
                        0,
                        0,
                        &cat(&[1u32.to_be_bytes().to_vec(), flac_entry(a)]),
                    ),
                    sample_tables(&audio_sizes, &audio_durations, audio_offset, None),
                ]),
            );
            let duration = (audio_total * u64::from(movie_scale) / u64::from(a.sample_rate)) as u32;
            parts.push(boxed(
                b"trak",
                &cat(&[
                    tkhd(2, duration, true, 0, 0),
                    boxed(
                        b"mdia",
                        &cat(&[
                            mdhd(a.sample_rate, audio_total as u32),
                            hdlr(b"soun", "orangu sound"),
                            boxed(b"minf", &cat(&[smhd, dinf(), audio_stbl])),
                        ]),
                    ),
                ]),
            ));
        }
        boxed(b"moov", &cat(&parts))
    };
    // The moov's size does not depend on the offsets (co64 is fixed
    // width), so measure it once with placeholders.
    let moov_len = build_moov(0, 0).len() as u64;
    let mdat_header = 16u64; // largesize form
    let video_offset = ftyp.len() as u64 + moov_len + mdat_header;
    let audio_offset = video_offset + video_bytes;
    let moov = build_moov(video_offset, audio_offset);
    let audio_bytes: u64 = audio_sizes.iter().map(|&s| s as u64).sum();
    let mdat_len = mdat_header + video_bytes + audio_bytes;
    let mut out = Vec::with_capacity((ftyp.len() as u64 + moov_len + mdat_len) as usize);
    out.extend(ftyp);
    out.extend(moov);
    out.extend(1u32.to_be_bytes());
    out.extend(b"mdat");
    out.extend(mdat_len.to_be_bytes());
    for s in &video.samples {
        out.extend(&s.0);
    }
    if let Some(a) = audio {
        for f in &a.frames {
            out.extend(&f.0);
        }
    }
    Ok(out)
}

/// Encodes a decoded clip and its sound into an MP4.
pub fn encode_clip(
    frames: &[Vec<f32>],
    width: usize,
    height: usize,
    fps: u32,
    audio: Option<(&[f32], usize, u32)>,
    settings: Av1Settings,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Vec<u8>> {
    let video = encode_av1(frames, width, height, fps, settings, cancel).context("encoding AV1")?;
    let flac = audio.map(|(samples, channels, rate)| super::flac::encode(samples, channels, rate));
    write(&video, fps, flac.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_is_mid_range_luma_and_neutral_chroma() {
        let [y, u, v] = rgb_to_yuv420(&[0.5; 2 * 2 * 3], 2, 2);
        assert!(y.iter().all(|&l| l == 126));
        assert_eq!(u, vec![128]);
        assert_eq!(v, vec![128]);
    }

    #[test]
    fn temporal_delimiters_are_dropped() {
        // A TD (type 2, size 0) then a padding OBU (type 15) of 1 byte.
        let tu = [0x12, 0x00, 0x7A, 0x01, 0xFF];
        assert_eq!(
            strip_temporal_delimiters(&tu).unwrap(),
            vec![0x7A, 0x01, 0xFF]
        );
    }

    #[test]
    #[ignore]
    fn a_clip_round_trips_through_ffprobe() {
        let (w, h, n) = (64usize, 48usize, 12usize);
        let frames: Vec<Vec<f32>> = (0..n)
            .map(|t| {
                (0..w * h)
                    .flat_map(|i| {
                        let x = (i % w) as f32 / w as f32;
                        [x, t as f32 / n as f32, 1.0 - x]
                    })
                    .collect()
            })
            .collect();
        let rate = 48000u32;
        let audio: Vec<f32> = (0..rate as usize / 2)
            .flat_map(|i| {
                let v = (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.3;
                [v, v]
            })
            .collect();
        let mp4 = encode_clip(
            &frames,
            w,
            h,
            24,
            Some((&audio, 2, rate)),
            Av1Settings::default(),
            None,
        )
        .unwrap();
        let path = std::env::var("ORANGU_VIDEO_PROBE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("orangu-video-probe.mp4"));
        std::fs::write(&path, &mp4).unwrap();
        eprintln!("wrote {} bytes to {}", mp4.len(), path.display());
    }
}
