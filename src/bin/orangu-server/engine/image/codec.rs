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

//! Pictures in and out of the pipeline: PNG, JPEG, GIF and WebP both ways,
//! SVG in.
//!
//! The pipeline works in the VAE's terms — an RGB `Feature` in `[-1, 1]`
//! — and everything a browser or an API client sends or expects is bytes
//! in a container. Both directions go through the `image` crate the tree
//! already carries (via `printpdf`): PNG and JPEG as it builds them there,
//! GIF (one frame, a 256-colour palette) and lossless WebP added; an SVG
//! is drawn with `resvg`, the same renderer the web console's diagrams
//! use, and then treated as the raster it became. An animated GIF or WebP
//! is read as its first frame.

use anyhow::{Context, Result, anyhow, bail, ensure};
use std::io::Cursor;

use super::vae::Feature;

/// A container the server can write: four rasters, and SVG — a document
/// that carries the picture as an embedded PNG, there being no
/// pixels-to-vector; what an `.svg` that any viewer opens can be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    #[default]
    Png,
    Jpeg,
    Gif,
    Webp,
    Svg,
}

impl ImageFormat {
    pub fn mime(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
            Self::Svg => "image/svg+xml",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Gif => "gif",
            Self::Webp => "webp",
            Self::Svg => "svg",
        }
    }

    /// The name the config key and `/props` use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Gif => "gif",
            Self::Webp => "webp",
            Self::Svg => "svg",
        }
    }

    /// The format an attachment's MIME type names, so a reply can come back
    /// in the same container the picture arrived in. `None` for anything
    /// that is not a picture this server writes.
    pub fn from_mime(mime: &str) -> Option<Self> {
        match mime.trim().to_ascii_lowercase().as_str() {
            "image/png" => Some(Self::Png),
            "image/jpeg" | "image/jpg" => Some(Self::Jpeg),
            "image/gif" => Some(Self::Gif),
            "image/webp" => Some(Self::Webp),
            "image/svg+xml" => Some(Self::Svg),
            _ => None,
        }
    }

    /// OpenAI's `output_format` spellings, plus the three more this server
    /// writes.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpeg" | "jpg" => Some(Self::Jpeg),
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::Webp),
            "svg" => Some(Self::Svg),
            _ => None,
        }
    }

    /// The extension a file name ends in, when it names a raster.
    pub fn from_extension(name: &str) -> Option<Self> {
        match name
            .rsplit('.')
            .next()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("png") => Some(Self::Png),
            Some("jpg") | Some("jpeg") => Some(Self::Jpeg),
            Some("gif") => Some(Self::Gif),
            Some("webp") => Some(Self::Webp),
            Some("svg") => Some(Self::Svg),
            _ => None,
        }
    }
}

/// Whether `mime` (or, failing that, `name`'s extension) is a picture this
/// module can read: the four rasters, or an SVG.
pub fn is_readable_image(name: &str, mime: &str) -> bool {
    let mime = mime.trim().to_ascii_lowercase();
    if mime == "image/svg+xml" || ImageFormat::from_mime(&mime).is_some() {
        return true;
    }
    let lower = name.to_ascii_lowercase();
    (mime.is_empty() || mime == "application/octet-stream")
        && (lower.ends_with(".svg") || ImageFormat::from_extension(&lower).is_some())
}

/// Decodes `bytes` — PNG, JPEG or SVG — and resizes to `width` x `height`,
/// returning RGB in `[-1, 1]`, channel-last. Alpha is composited on white,
/// which is what a browser shows for a transparent picture on this
/// console's light theme and the only fixed choice that is never black.
pub fn decode_to_feature(bytes: &[u8], width: usize, height: usize) -> Result<Feature> {
    let resized = decode_resized(bytes, width, height)?;
    let mut data = Vec::with_capacity(width * height * 3);
    for px in resized.pixels() {
        let a = px[3] as f32 / 255.0;
        for c in 0..3 {
            let v = px[c] as f32 / 255.0 * a + (1.0 - a);
            data.push(v * 2.0 - 1.0);
        }
    }
    Ok(Feature::new(height, width, 3, data))
}

/// [`decode_to_feature`] keeping the alpha channel: RGBA in `[-1, 1]`, a
/// picture without one fully opaque — what the Qwen-Image 2.1 VAE reads
/// (diffusers converts every picture to `RGBA` before encoding it).
pub fn decode_to_rgba_feature(bytes: &[u8], width: usize, height: usize) -> Result<Feature> {
    let resized = decode_resized(bytes, width, height)?;
    let data = resized
        .as_raw()
        .iter()
        .map(|&v| v as f32 / 255.0 * 2.0 - 1.0)
        .collect();
    Ok(Feature::new(height, width, 4, data))
}

fn decode_resized(bytes: &[u8], width: usize, height: usize) -> Result<image::RgbaImage> {
    let rgba = if looks_like_svg(bytes) {
        rasterize_svg(bytes)?
    } else {
        image::load_from_memory(bytes)
            .context("decoding the picture (PNG or JPEG)")?
            .into_rgba8()
    };
    let resized = if rgba.width() as usize == width && rgba.height() as usize == height {
        rgba
    } else {
        image::imageops::resize(
            &rgba,
            width as u32,
            height as u32,
            image::imageops::FilterType::Lanczos3,
        )
    };
    Ok(resized)
}

/// The pixel size of a picture without decoding all of it, for choosing a
/// generation size from an attachment.
pub fn dimensions(bytes: &[u8]) -> Result<(usize, usize)> {
    if looks_like_svg(bytes) {
        let tree = svg_tree(bytes)?;
        let size = tree.size().to_int_size();
        return Ok((size.width() as usize, size.height() as usize));
    }
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("reading the picture's header")?;
    let (w, h) = reader
        .into_dimensions()
        .context("reading the picture's size")?;
    Ok((w as usize, h as usize))
}

fn looks_like_svg(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(512)];
    let text = String::from_utf8_lossy(head);
    text.trim_start().starts_with('<') && text.contains("<svg")
}

fn svg_tree(bytes: &[u8]) -> Result<resvg::usvg::Tree> {
    let options = resvg::usvg::Options::default();
    resvg::usvg::Tree::from_data(bytes, &options).map_err(|err| anyhow!("parsing the SVG: {err}"))
}

fn rasterize_svg(bytes: &[u8]) -> Result<image::RgbaImage> {
    let tree = svg_tree(bytes)?;
    let size = tree.size().to_int_size();
    if u64::from(size.width()) * u64::from(size.height()) > 16_777_216 {
        bail!("the SVG is larger than 16 megapixels");
    }
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size.width(), size.height())
        .ok_or_else(|| anyhow!("the SVG has no size"))?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::default(),
        &mut pixmap.as_mut(),
    );
    // tiny-skia stores premultiplied RGBA; `image` wants straight alpha.
    let mut data = Vec::with_capacity(pixmap.data().len());
    for px in pixmap.pixels() {
        let c = px.demultiply();
        data.extend_from_slice(&[c.red(), c.green(), c.blue(), c.alpha()]);
    }
    image::RgbaImage::from_raw(size.width(), size.height(), data)
        .ok_or_else(|| anyhow!("the rasterized SVG has the wrong size"))
}

/// Encodes an RGB or RGBA `[-1, 1]` feature as `format`. Alpha is kept by
/// every format that has it (PNG, GIF, WebP, and SVG's embedded PNG) and
/// composited on white for JPEG, which has none.
pub fn encode(picture: &Feature, format: ImageFormat) -> Result<Vec<u8>> {
    ensure!(
        picture.channels == 3 || picture.channels == 4,
        "a picture has 3 or 4 channels, not {}",
        picture.channels
    );
    let pixels: Vec<u8> = picture
        .data
        .iter()
        .map(|v| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let (w, h) = (picture.width as u32, picture.height as u32);
    let img = if picture.channels == 4 {
        image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(w, h, pixels)
                .ok_or_else(|| anyhow!("the picture's data does not match its size"))?,
        )
    } else {
        image::DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(w, h, pixels)
                .ok_or_else(|| anyhow!("the picture's data does not match its size"))?,
        )
    };
    let mut out = Cursor::new(Vec::new());
    match format {
        ImageFormat::Png => img
            .write_to(&mut out, image::ImageFormat::Png)
            .context("encoding PNG")?,
        ImageFormat::Jpeg => {
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 92);
            encoder
                .encode_image(&on_white(&img))
                .context("encoding JPEG")?;
        }
        // One frame on a 256-colour palette: GIF's own limit, so a
        // photograph comes back posterized. It is what was asked for.
        ImageFormat::Gif => {
            let mut encoder = image::codecs::gif::GifEncoder::new(&mut out);
            encoder
                .encode_frame(image::Frame::new(img.to_rgba8()))
                .context("encoding GIF")?;
        }
        // Lossless: the `image` crate writes no lossy WebP, and a picture
        // that took minutes to make deserves its pixels back.
        ImageFormat::Webp => {
            use image::ImageEncoder as _;
            let encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut out);
            let color = if picture.channels == 4 {
                image::ExtendedColorType::Rgba8
            } else {
                image::ExtendedColorType::Rgb8
            };
            encoder
                .write_image(img.as_bytes(), w, h, color)
                .context("encoding WebP")?;
        }
        // The picture as PNG inside an SVG element of its own size — a
        // document, not a drawing, but one every SVG viewer and browser
        // opens, and the format that was asked for.
        ImageFormat::Svg => {
            let mut png = Cursor::new(Vec::new());
            img.write_to(&mut png, image::ImageFormat::Png)
                .context("encoding PNG")?;
            use base64::Engine as _;
            let data = base64::engine::general_purpose::STANDARD.encode(png.into_inner());
            let svg = format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
                 viewBox=\"0 0 {w} {h}\"><image width=\"{w}\" height=\"{h}\" \
                 href=\"data:image/png;base64,{data}\"/></svg>\n"
            );
            return Ok(svg.into_bytes());
        }
    }
    Ok(out.into_inner())
}

/// A picture's RGB with its alpha composited on white — the JPEG a
/// transparent picture becomes, as [`decode_to_feature`] reads one.
fn on_white(img: &image::DynamicImage) -> image::RgbImage {
    let rgba = img.to_rgba8();
    let mut rgb = image::RgbImage::new(rgba.width(), rgba.height());
    for (out, px) in rgb.pixels_mut().zip(rgba.pixels()) {
        let a = px[3] as f32 / 255.0;
        for c in 0..3 {
            out[c] = (px[c] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        }
    }
    rgb
}

/// A generation size from a picture's size: the same aspect ratio, scaled
/// down so the longer side is at most `max_side` (never up — a small
/// picture is redrawn at its own size), and both sides multiples of `unit`
/// — the pixels one latent token covers along a side, 16 for Qwen-Image
/// (the VAE's 8 times the transformer's 2x2 patch) and 32 for Qwen-Image
/// 2.1 (see `Pipeline::size_unit`).
pub fn fit_generation_size(
    (width, height): (usize, usize),
    max_side: usize,
    unit: usize,
) -> (usize, usize) {
    let longest = width.max(height).max(1) as f64;
    let scale = (max_side as f64 / longest).min(1.0);
    let round = |v: f64| ((v * scale / unit as f64).round().max(1.0) as usize) * unit;
    (round(width as f64), round(height as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_and_jpeg_round_trip_through_the_feature_layout() {
        // A 16x16 gradient: red across, green down, in [-1, 1].
        let mut data = Vec::new();
        for y in 0..16 {
            for x in 0..16 {
                data.push(x as f32 / 15.0 * 2.0 - 1.0);
                data.push(y as f32 / 15.0 * 2.0 - 1.0);
                data.push(-1.0);
            }
        }
        let feature = Feature::new(16, 16, 3, data);
        for format in [
            ImageFormat::Png,
            ImageFormat::Jpeg,
            ImageFormat::Gif,
            ImageFormat::Webp,
            ImageFormat::Svg,
        ] {
            let bytes = encode(&feature, format).unwrap();
            assert_eq!(dimensions(&bytes).unwrap(), (16, 16), "{format:?}");
            let back = decode_to_feature(&bytes, 16, 16).unwrap();
            assert_eq!((back.height, back.width, back.channels), (16, 16, 3));
            let worst = back
                .data
                .iter()
                .zip(&feature.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // PNG and lossless WebP (and SVG, PNG inside) are exact to
            // quantization; JPEG is lossy but close; GIF's 256-colour
            // palette is coarser on a gradient.
            let tolerance = match format {
                ImageFormat::Png | ImageFormat::Webp | ImageFormat::Svg => 0.01,
                ImageFormat::Jpeg => 0.15,
                ImageFormat::Gif => 0.25,
            };
            assert!(worst < tolerance, "{format:?}: worst error {worst}");
        }
        assert!(
            encode(&feature, ImageFormat::Png)
                .unwrap()
                .starts_with(b"\x89PNG")
        );
        assert!(
            encode(&feature, ImageFormat::Jpeg)
                .unwrap()
                .starts_with(&[0xFF, 0xD8])
        );
        assert!(
            encode(&feature, ImageFormat::Gif)
                .unwrap()
                .starts_with(b"GIF8")
        );
        assert!(
            encode(&feature, ImageFormat::Webp)
                .unwrap()
                .starts_with(b"RIFF")
        );
    }

    /// A picture with transparency — what Qwen-Image 2.1 draws — keeps its
    /// alpha through PNG and WebP, and JPEG shows it on white.
    #[test]
    fn an_rgba_picture_keeps_its_alpha_where_the_format_has_one() {
        // Left half opaque black, right half fully transparent black.
        let mut data = Vec::new();
        for _y in 0..8 {
            for x in 0..8 {
                data.extend_from_slice(&[-1.0, -1.0, -1.0, if x < 4 { 1.0 } else { -1.0 }]);
            }
        }
        let feature = Feature::new(8, 8, 4, data);
        for format in [ImageFormat::Png, ImageFormat::Webp] {
            let bytes = encode(&feature, format).unwrap();
            let back = decode_to_rgba_feature(&bytes, 8, 8).unwrap();
            assert_eq!(back.data, feature.data, "{format:?}");
            // Composited, the transparent half is white.
            let rgb = decode_to_feature(&bytes, 8, 8).unwrap();
            assert_eq!(&rgb.data[..3], &[-1.0, -1.0, -1.0]);
            assert_eq!(&rgb.data[7 * 3..8 * 3], &[1.0, 1.0, 1.0]);
        }
        let jpeg = decode_to_feature(&encode(&feature, ImageFormat::Jpeg).unwrap(), 8, 8).unwrap();
        assert!(
            jpeg.data[7 * 3] > 0.8,
            "the transparent half is white in a JPEG"
        );
        // A picture without alpha reads back fully opaque.
        let opaque = Feature::new(1, 1, 3, vec![0.0, 0.0, 0.0]);
        let back =
            decode_to_rgba_feature(&encode(&opaque, ImageFormat::Png).unwrap(), 1, 1).unwrap();
        assert_eq!(back.data[3], 1.0);
    }

    #[test]
    fn an_svg_is_drawn_then_read_like_a_raster() {
        let svg = b"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"8\" height=\"8\">\
                    <rect width=\"8\" height=\"8\" fill=\"#ff0000\"/></svg>";
        assert_eq!(dimensions(svg).unwrap(), (8, 8));
        let f = decode_to_feature(svg, 8, 8).unwrap();
        assert!((f.data[0] - 1.0).abs() < 0.02, "red {}", f.data[0]);
        assert!((f.data[1] + 1.0).abs() < 0.02, "green {}", f.data[1]);
        assert!((f.data[2] + 1.0).abs() < 0.02, "blue {}", f.data[2]);
    }

    /// An SVG result is a document of the picture's size with the pixels
    /// inside as PNG — and reads back as the same picture, so a reply in
    /// SVG can be attached to the next turn like any other.
    #[test]
    fn an_svg_result_carries_the_picture_and_reads_back() {
        let f = Feature::new(8, 8, 3, vec![1.0; 8 * 8 * 3]);
        let svg = encode(&f, ImageFormat::Svg).unwrap();
        let text = std::str::from_utf8(&svg).unwrap();
        assert!(
            text.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"8\" height=\"8\"")
        );
        assert!(text.contains("href=\"data:image/png;base64,"));
        assert_eq!(dimensions(&svg).unwrap(), (8, 8));
        let back = decode_to_feature(&svg, 8, 8).unwrap();
        assert!((back.data[0] - 1.0).abs() < 0.02, "red {}", back.data[0]);
        assert_eq!(ImageFormat::parse("svg"), Some(ImageFormat::Svg));
        assert_eq!(
            ImageFormat::from_mime("image/svg+xml"),
            Some(ImageFormat::Svg)
        );
        assert_eq!(ImageFormat::Svg.extension(), "svg");
    }

    #[test]
    fn decoding_resizes_to_the_requested_size() {
        let feature = Feature::new(4, 4, 3, vec![0.0; 48]);
        let bytes = encode(&feature, ImageFormat::Png).unwrap();
        let back = decode_to_feature(&bytes, 8, 16).unwrap();
        assert_eq!((back.width, back.height), (8, 16));
    }

    #[test]
    fn generation_sizes_keep_the_aspect_and_snap_to_sixteen() {
        assert_eq!(fit_generation_size((1024, 1024), 1024, 16), (1024, 1024));
        assert_eq!(fit_generation_size((4000, 3000), 1024, 16), (1024, 768));
        assert_eq!(fit_generation_size((300, 200), 1024, 16), (304, 208));
        assert_eq!(fit_generation_size((1, 1), 1024, 16), (16, 16));
        // Qwen-Image 2.1's token is 32 pixels on a side.
        assert_eq!(fit_generation_size((300, 200), 1024, 32), (288, 192));
        assert_eq!(fit_generation_size((1, 1), 1024, 32), (32, 32));
    }

    #[test]
    fn formats_are_named_both_ways() {
        assert_eq!(
            ImageFormat::from_mime("image/jpeg"),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(
            ImageFormat::from_mime("image/svg+xml"),
            Some(ImageFormat::Svg)
        );
        assert_eq!(ImageFormat::parse("PNG"), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::parse("webp"), Some(ImageFormat::Webp));
        assert_eq!(ImageFormat::from_extension("A.GIF"), Some(ImageFormat::Gif));
        assert!(is_readable_image("a.svg", ""));
        assert!(is_readable_image("photo", "image/jpeg"));
        assert!(is_readable_image("a.gif", "image/gif"));
        assert!(is_readable_image("a.webp", "application/octet-stream"));
        assert!(!is_readable_image("a.bmp", "image/bmp"));
    }
}
