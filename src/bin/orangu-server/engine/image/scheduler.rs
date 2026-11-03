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

//! The flow-matching sampler Qwen-Image is trained against — diffusers'
//! `FlowMatchEulerDiscreteScheduler` with the configuration the released
//! checkpoint carries (`scheduler/scheduler_config.json`), plus the seeded
//! Gaussian noise the latents start from.
//!
//! A flow model predicts a *velocity*: at noise level `sigma` the latent is
//! `x = (1 - sigma) * clean + sigma * noise`, and the model's output is the
//! direction from noise towards the clean image. One Euler step moves the
//! latent along it by the distance to the next noise level, and the last
//! level is `0`. Everything else here is deciding *which* noise levels to
//! visit: the schedule is shifted towards the noisy end by an amount that
//! grows with the image's token count (`use_dynamic_shifting`), and then
//! stretched so it ends at `shift_terminal` rather than reaching zero
//! before the final step.

/// The checkpoint's scheduler configuration, verbatim.
#[derive(Debug, Clone, Copy)]
pub struct ScheduleConfig {
    pub base_image_seq_len: f64,
    pub max_image_seq_len: f64,
    pub base_shift: f64,
    pub max_shift: f64,
    pub shift_terminal: f64,
}

impl ScheduleConfig {
    /// `Qwen/Qwen-Image-2512`'s `scheduler_config.json`.
    pub const QWEN_IMAGE: Self = Self {
        base_image_seq_len: 256.0,
        max_image_seq_len: 8192.0,
        base_shift: 0.5,
        max_shift: 0.9,
        shift_terminal: 0.02,
    };
}

/// The noise levels one generation visits — `steps + 1` values, the first
/// being the level the initial noise is at and the last `0`.
///
/// `image_seq_len` is the number of image tokens (`H/16 * W/16`): a larger
/// image gets its schedule shifted further towards the noisy end, which is
/// what `use_dynamic_shifting` means.
pub fn sigmas(config: &ScheduleConfig, steps: usize, image_seq_len: usize) -> Vec<f32> {
    assert!(steps > 0, "a schedule needs at least one step");
    // diffusers: `np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)`.
    let raw: Vec<f64> = (0..steps)
        .map(|i| {
            if steps == 1 {
                1.0
            } else {
                let last = 1.0 / steps as f64;
                1.0 + (last - 1.0) * (i as f64 / (steps - 1) as f64)
            }
        })
        .collect();
    let mu = calculate_shift(config, image_seq_len as f64);
    // `time_shift`, exponential: `exp(mu) / (exp(mu) + (1/t - 1))`.
    let shifted: Vec<f64> = raw
        .iter()
        .map(|&t| mu.exp() / (mu.exp() + (1.0 / t - 1.0)))
        .collect();
    // `stretch_shift_to_terminal`: the last level lands on `shift_terminal`.
    let one_minus_last = 1.0 - shifted[shifted.len() - 1];
    let scale_factor = one_minus_last / (1.0 - config.shift_terminal);
    let mut out: Vec<f32> = shifted
        .iter()
        .map(|&s| (1.0 - (1.0 - s) / scale_factor) as f32)
        .collect();
    out.push(0.0);
    out
}

/// FLUX.2's noise levels — `steps + 1` values, as [`sigmas`] returns:
/// the same linear spacing and exponential shift, but with the shift `mu`
/// fitted to both the token count and the step count
/// ([`flux2_empirical_mu`]) and no stretch to a terminal level
/// (`shift_terminal` is unset in its `scheduler_config.json`).
pub fn flux2_sigmas(steps: usize, image_seq_len: usize) -> Vec<f32> {
    assert!(steps > 0, "a schedule needs at least one step");
    let mu = flux2_empirical_mu(image_seq_len, steps).exp();
    let mut out: Vec<f32> = (0..steps)
        .map(|i| {
            let t = if steps == 1 {
                1.0
            } else {
                1.0 + (1.0 / steps as f64 - 1.0) * (i as f64 / (steps - 1) as f64)
            };
            (mu / (mu + (1.0 / t - 1.0))) as f32
        })
        .collect();
    out.push(0.0);
    out
}

/// MOSS-Image's noise levels — `MOSSImageScheduler.set_timesteps`: `steps`
/// levels spaced linearly from 1 and shifted by `shift / (1 + (shift - 1)
/// s)` with the release shift of 5, each then snapped to the nearest of the
/// thousand levels the model was trained at, consecutive repeats dropped,
/// and the terminal 0 appended. Snapping can merge levels at large step
/// counts, so the schedule may hold fewer than `steps + 1` values.
pub fn moss_sigmas(steps: usize) -> Vec<f32> {
    assert!(steps > 0, "a schedule needs at least one step");
    const TOTAL: usize = 1000;
    const SHIFT: f64 = 5.0;
    // torch.linspace in single precision: the first half counted up from
    // `start`, the second down from `end`.
    let linspace = |start: f32, end: f32, count: usize| -> Vec<f32> {
        if count == 1 {
            return vec![start];
        }
        let step = (end - start) / (count - 1) as f32;
        (0..count)
            .map(|i| {
                if i < count / 2 {
                    start + step * i as f32
                } else {
                    end - step * (count - 1 - i) as f32
                }
            })
            .collect()
    };
    let grid = |count: usize, shift: f64| -> Vec<f32> {
        let lower = (shift / (TOTAL as f64 + shift - 1.0)) as f32;
        let shift = shift as f32;
        linspace(1.0, lower, count)
            .into_iter()
            .map(|v| shift * v / (1.0 + (shift - 1.0) * v))
            .collect()
    };
    let mut training = grid(TOTAL, 1.0);
    training.reverse();
    let mut out: Vec<f32> = Vec::with_capacity(steps + 1);
    for s in grid(steps, SHIFT) {
        let upper = training.partition_point(|&t| t < s).min(TOTAL - 1);
        let lower = upper.saturating_sub(1);
        let nearest = if (s - training[lower]).abs() <= (training[upper] - s).abs() {
            lower
        } else {
            upper
        };
        let level = training[nearest];
        if out.last() != Some(&level) {
            out.push(level);
        }
    }
    out.push(0.0);
    out
}

/// diffusers' `compute_empirical_mu` for FLUX.2: two lines in the token
/// count, for 10 and 200 steps, interpolated linearly in the step count;
/// past 4300 tokens the 200-step line alone.
pub fn flux2_empirical_mu(image_seq_len: usize, steps: usize) -> f64 {
    let (a1, b1) = (8.738_095_24e-5, 1.898_333_33);
    let (a2, b2) = (0.000_169_27, 0.456_666_66);
    let len = image_seq_len as f64;
    if image_seq_len > 4300 {
        return a2 * len + b2;
    }
    let m_200 = a2 * len + b2;
    let m_10 = a1 * len + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * steps as f64 + b
}

/// The eight-step schedule `Qwen/Qwen-Image-2.1-Turbo` was distilled for
/// and ships with — `steps + 1` levels, as [`sigmas`] returns.
pub const QWEN_IMAGE_21_TURBO: [f32; 9] = [
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0,
];

/// A fixed schedule's levels for `steps` steps: the table itself at its
/// own count, and otherwise the table read at `steps + 1` evenly spaced
/// positions, linearly between its entries — so the start is still pure
/// noise, the end still `0`, and the shape the checkpoint was trained on
/// is kept.
pub fn resample(table: &[f32], steps: usize) -> Vec<f32> {
    assert!(steps > 0, "a schedule needs at least one step");
    assert!(table.len() >= 2, "a schedule table needs two levels");
    if table.len() == steps + 1 {
        return table.to_vec();
    }
    let last = (table.len() - 1) as f64;
    (0..=steps)
        .map(|i| {
            let at = last * i as f64 / steps as f64;
            let lo = (at.floor() as usize).min(table.len() - 2);
            let frac = at - lo as f64;
            (table[lo] as f64 + (table[lo + 1] as f64 - table[lo] as f64) * frac) as f32
        })
        .collect()
}

/// diffusers' `calculate_shift`: the shift grows linearly with the token
/// count between the two anchor points the checkpoint names.
fn calculate_shift(config: &ScheduleConfig, image_seq_len: f64) -> f64 {
    let m = (config.max_shift - config.base_shift)
        / (config.max_image_seq_len - config.base_image_seq_len);
    let b = config.base_shift - m * config.base_image_seq_len;
    image_seq_len * m + b
}

/// One Euler step: `x += (sigma_next - sigma) * velocity`, in place.
pub fn euler_step(latents: &mut [f32], velocity: &[f32], sigma: f32, sigma_next: f32) {
    debug_assert_eq!(latents.len(), velocity.len());
    let dt = sigma_next - sigma;
    for (x, v) in latents.iter_mut().zip(velocity) {
        *x += dt * v;
    }
}

/// diffusers' `scale_noise`: the latent an image-to-image run starts from,
/// `sigma * noise + (1 - sigma) * clean`.
pub fn scale_noise(clean: &[f32], noise: &[f32], sigma: f32) -> Vec<f32> {
    debug_assert_eq!(clean.len(), noise.len());
    clean
        .iter()
        .zip(noise)
        .map(|(c, n)| sigma * n + (1.0 - sigma) * c)
        .collect()
}

/// Where an image-to-image run enters the schedule: diffusers'
/// `get_timesteps` — `strength` of the steps are run, the first
/// `steps - round_down(steps * strength)` skipped.
pub fn first_step_for_strength(steps: usize, strength: f32) -> usize {
    let init = (steps as f32 * strength.clamp(0.0, 1.0)).min(steps as f32);
    (steps as f32 - init).max(0.0) as usize
}

/// A seeded source of standard-normal samples: the initial latent noise, so
/// a `seed` reproduces an image on the same build.
///
/// A small xoshiro256** over Box–Muller rather than a `rand` distribution
/// crate: the point is a fixed, dependency-free sequence per seed, and one
/// that is the same on every platform this runs on.
pub struct Noise {
    state: [u64; 4],
    spare: Option<f32>,
}

impl Noise {
    pub fn seeded(seed: u64) -> Self {
        // SplitMix64 to spread the seed over the four words, as xoshiro's
        // authors recommend for seeding from a single integer.
        let mut z = seed;
        let mut state = [0u64; 4];
        for word in state.iter_mut() {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut s = z;
            s = (s ^ (s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            s = (s ^ (s >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *word = s ^ (s >> 31);
        }
        Self { state, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `(0, 1]` — never `0`, so the logarithm below is finite.
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }

    pub fn next_normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some((r * theta.sin()) as f32);
        (r * theta.cos()) as f32
    }

    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_normal()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against `MOSSImageScheduler.set_timesteps` worked in NumPy: fifty
    /// steps keep fifty distinct training levels; five hundred collapse to
    /// 415 where the shifted spacing is finer than the grid.
    #[test]
    fn moss_levels_are_shifted_and_snapped_to_the_training_grid() {
        let s = moss_sigmas(50);
        assert_eq!(s.len(), 51);
        let near = |a: f32, b: f32| (a - b).abs() < 2e-6;
        assert!(near(s[0], 1.0) && near(s[1], 0.996) && near(s[2], 0.992));
        assert!(near(s[24], 0.84) && near(s[25], 0.829));
        assert!(near(s[47], 0.193) && near(s[48], 0.115) && near(s[49], 0.024));
        assert_eq!(s[50], 0.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
        assert_eq!(moss_sigmas(500).len(), 416);
    }

    /// Against diffusers' `compute_empirical_mu` and the shifted
    /// `np.linspace` it feeds, worked in Python.
    #[test]
    fn flux2_schedule_matches_the_reference() {
        assert!((flux2_empirical_mu(4096, 50) - 2.023_351_157).abs() < 1e-6);
        assert!((flux2_empirical_mu(5000, 20) - 1.303_016_66).abs() < 1e-6);
        let s = flux2_sigmas(50, 4096);
        assert_eq!(s.len(), 51);
        assert_eq!((s[0], s[50]), (1.0, 0.0));
        assert!((s[1] - 0.997_309).abs() < 1e-6);
        assert!((s[49] - 0.133_719).abs() < 1e-6);
        let s = flux2_sigmas(4, 1024);
        assert!((s[1] - 0.958_085).abs() < 1e-6 && (s[3] - 0.717_497).abs() < 1e-6);
    }

    /// The Turbo table is served as-is at its own eight steps, and read
    /// between its entries at any other count, still from noise to `0`.
    #[test]
    fn a_fixed_schedule_resamples_between_its_levels() {
        assert_eq!(resample(&QWEN_IMAGE_21_TURBO, 8), QWEN_IMAGE_21_TURBO);
        let four = resample(&QWEN_IMAGE_21_TURBO, 4);
        assert_eq!(four, vec![1.0, 0.95418, 0.89508, 0.704534, 0.0]);
        let sixteen = resample(&QWEN_IMAGE_21_TURBO, 16);
        assert_eq!(sixteen.len(), 17);
        assert_eq!((sixteen[0], sixteen[16]), (1.0, 0.0));
        assert_eq!(sixteen[2], QWEN_IMAGE_21_TURBO[1]);
        assert!((sixteen[1] - (1.0 + 0.978453) / 2.0).abs() < 1e-6);
        assert!(sixteen.windows(2).all(|w| w[0] > w[1]));
    }

    /// The values diffusers computes for a 1024x1024 image at 4 steps
    /// (`image_seq_len = 4096`, `mu = 0.8677`), from
    /// `FlowMatchEulerDiscreteScheduler.set_timesteps(4, sigmas=linspace(1,
    /// 0.25, 4), mu=...)` with Qwen-Image's config: the terminal stretch is
    /// what makes the last non-zero sigma exactly `shift_terminal`.
    #[test]
    fn the_schedule_matches_diffusers_at_1024_square() {
        let s = sigmas(&ScheduleConfig::QWEN_IMAGE, 4, 4096);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 1.0).abs() < 1e-6, "{s:?}");
        // mu = 0.5 + (0.9-0.5)/(8192-256) * (4096-256) = 0.693..; the raw
        // levels 0.75/0.5/0.25 shift to exp(mu)/(exp(mu)+1/t-1) and the whole
        // set is then stretched so the last one is 0.02.
        let mu: f64 = 0.5 + (0.9 - 0.5) / (8192.0 - 256.0) * (4096.0 - 256.0);
        let shift = |t: f64| mu.exp() / (mu.exp() + (1.0 / t - 1.0));
        let last = shift(0.25);
        let factor = (1.0 - last) / (1.0 - 0.02);
        let expect = |t: f64| (1.0 - (1.0 - shift(t)) / factor) as f32;
        assert!((s[1] - expect(0.75)).abs() < 1e-6, "{s:?}");
        assert!((s[2] - expect(0.5)).abs() < 1e-6, "{s:?}");
        assert!((s[3] - 0.02).abs() < 1e-6, "{s:?}");
        assert_eq!(s[4], 0.0);
    }

    #[test]
    fn a_larger_image_shifts_the_schedule_towards_noise() {
        let small = sigmas(&ScheduleConfig::QWEN_IMAGE, 10, 256);
        let large = sigmas(&ScheduleConfig::QWEN_IMAGE, 10, 8192);
        // Same endpoints, every middle level higher for the larger image.
        assert!((small[0] - large[0]).abs() < 1e-6);
        for i in 1..9 {
            assert!(
                large[i] > small[i],
                "step {i}: {} vs {}",
                large[i],
                small[i]
            );
        }
    }

    #[test]
    fn one_euler_step_to_zero_recovers_the_clean_latent() {
        // With sigma = 1 the latent is pure noise. The flow's velocity is
        // `d x / d sigma = noise - clean` (what the model is trained to
        // predict), so one step from sigma 1 to 0 must land exactly on
        // `clean`.
        let clean = [0.5f32, -1.0, 2.0];
        let noise = [1.0f32, 1.0, 1.0];
        let mut x = scale_noise(&clean, &noise, 1.0);
        let velocity: Vec<f32> = clean.iter().zip(&noise).map(|(c, n)| n - c).collect();
        euler_step(&mut x, &velocity, 1.0, 0.0);
        for (a, b) in x.iter().zip(&clean) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn strength_picks_where_image_to_image_enters() {
        assert_eq!(first_step_for_strength(20, 1.0), 0);
        assert_eq!(first_step_for_strength(20, 0.6), 8);
        assert_eq!(first_step_for_strength(20, 0.0), 20);
    }

    #[test]
    fn seeded_noise_is_reproducible_and_roughly_standard_normal() {
        let a = Noise::seeded(7).normals(20_000);
        let b = Noise::seeded(7).normals(20_000);
        assert_eq!(a, b);
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
        assert_ne!(Noise::seeded(8).normals(4), Noise::seeded(7).normals(4));
    }
}
