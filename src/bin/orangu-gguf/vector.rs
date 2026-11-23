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

//! Eight `f32` lanes: one AVX2 register on x86_64 built with AVX2 and FMA,
//! two NEON registers on aarch64, an array elsewhere.
//!
//! Every operation follows one recipe on all three — fused multiply-adds,
//! the same `exp` polynomial, IEEE division, sums folded in one fixed order —
//! so a kernel written over [`V8`] gives the same bits on every machine.
//! The slice kernels below take any length: a tail shorter than eight lanes
//! is padded, run through the same lanes, and written back.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
mod lanes {
    use std::arch::x86_64::*;

    pub type Inner = __m256;

    // SAFETY (every function here): the build enables AVX2 and FMA, and
    // the pointer functions read or write eight floats the caller owns.
    #[inline(always)]
    pub fn splat(v: f32) -> Inner {
        unsafe { _mm256_set1_ps(v) }
    }
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> Inner {
        unsafe { _mm256_loadu_ps(p) }
    }
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: Inner) {
        unsafe { _mm256_storeu_ps(p, v) }
    }
    #[inline(always)]
    pub fn add(a: Inner, b: Inner) -> Inner {
        unsafe { _mm256_add_ps(a, b) }
    }
    #[inline(always)]
    pub fn sub(a: Inner, b: Inner) -> Inner {
        unsafe { _mm256_sub_ps(a, b) }
    }
    #[inline(always)]
    pub fn mul(a: Inner, b: Inner) -> Inner {
        unsafe { _mm256_mul_ps(a, b) }
    }
    #[inline(always)]
    pub fn div(a: Inner, b: Inner) -> Inner {
        unsafe { _mm256_div_ps(a, b) }
    }
    #[inline(always)]
    pub fn mul_add(a: Inner, b: Inner, c: Inner) -> Inner {
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }
    #[inline(always)]
    pub fn exp(x: Inner) -> Inner {
        unsafe { orangu::engine::tensor::exp_avx2(x) }
    }
    #[inline(always)]
    pub fn sums8(v: [Inner; 8]) -> Inner {
        unsafe {
            let a = _mm256_hadd_ps(v[0], v[1]);
            let b = _mm256_hadd_ps(v[2], v[3]);
            let c = _mm256_hadd_ps(v[4], v[5]);
            let d = _mm256_hadd_ps(v[6], v[7]);
            let e = _mm256_hadd_ps(a, b);
            let f = _mm256_hadd_ps(c, d);
            _mm256_add_ps(
                _mm256_permute2f128_ps(e, f, 0x20),
                _mm256_permute2f128_ps(e, f, 0x31),
            )
        }
    }
    #[inline(always)]
    pub fn lanes(v: Inner) -> [f32; 8] {
        let mut out = [0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) };
        out
    }
}

#[cfg(target_arch = "aarch64")]
mod lanes {
    use std::arch::aarch64::*;

    pub type Inner = (float32x4_t, float32x4_t);

    // SAFETY (every function here): NEON is baseline on aarch64, and the
    // pointer functions read or write eight floats the caller owns.
    #[inline(always)]
    pub fn splat(v: f32) -> Inner {
        unsafe { (vdupq_n_f32(v), vdupq_n_f32(v)) }
    }
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> Inner {
        unsafe { (vld1q_f32(p), vld1q_f32(p.add(4))) }
    }
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: Inner) {
        unsafe {
            vst1q_f32(p, v.0);
            vst1q_f32(p.add(4), v.1);
        }
    }
    #[inline(always)]
    pub fn add(a: Inner, b: Inner) -> Inner {
        unsafe { (vaddq_f32(a.0, b.0), vaddq_f32(a.1, b.1)) }
    }
    #[inline(always)]
    pub fn sub(a: Inner, b: Inner) -> Inner {
        unsafe { (vsubq_f32(a.0, b.0), vsubq_f32(a.1, b.1)) }
    }
    #[inline(always)]
    pub fn mul(a: Inner, b: Inner) -> Inner {
        unsafe { (vmulq_f32(a.0, b.0), vmulq_f32(a.1, b.1)) }
    }
    #[inline(always)]
    pub fn div(a: Inner, b: Inner) -> Inner {
        unsafe { (vdivq_f32(a.0, b.0), vdivq_f32(a.1, b.1)) }
    }
    #[inline(always)]
    pub fn mul_add(a: Inner, b: Inner, c: Inner) -> Inner {
        unsafe { (vfmaq_f32(c.0, a.0, b.0), vfmaq_f32(c.1, a.1, b.1)) }
    }
    #[inline(always)]
    pub fn exp(x: Inner) -> Inner {
        use orangu::engine::tensor::exp_neon;
        unsafe { (exp_neon(x.0), exp_neon(x.1)) }
    }
    #[inline(always)]
    pub fn sums8(v: [Inner; 8]) -> Inner {
        unsafe {
            let half = |a: &[Inner], hi: bool| {
                let p = |x: &Inner| if hi { x.1 } else { x.0 };
                vpaddq_f32(
                    vpaddq_f32(p(&a[0]), p(&a[1])),
                    vpaddq_f32(p(&a[2]), p(&a[3])),
                )
            };
            (
                vaddq_f32(half(&v[..4], false), half(&v[..4], true)),
                vaddq_f32(half(&v[4..], false), half(&v[4..], true)),
            )
        }
    }
    #[inline(always)]
    pub fn lanes(v: Inner) -> [f32; 8] {
        let mut out = [0f32; 8];
        unsafe { store(out.as_mut_ptr(), v) };
        out
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )
)))]
mod lanes {
    pub type Inner = [f32; 8];

    #[inline(always)]
    fn each(a: Inner, f: impl Fn(f32) -> f32) -> Inner {
        a.map(f)
    }
    #[inline(always)]
    fn zip(a: Inner, b: Inner, f: impl Fn(f32, f32) -> f32) -> Inner {
        std::array::from_fn(|i| f(a[i], b[i]))
    }
    #[inline(always)]
    pub fn splat(v: f32) -> Inner {
        [v; 8]
    }
    /// # Safety
    /// `p` points at eight readable floats.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> Inner {
        unsafe { std::ptr::read_unaligned(p.cast()) }
    }
    /// # Safety
    /// `p` points at eight writable floats.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: Inner) {
        unsafe { std::ptr::write_unaligned(p.cast(), v) }
    }
    #[inline(always)]
    pub fn add(a: Inner, b: Inner) -> Inner {
        zip(a, b, |x, y| x + y)
    }
    #[inline(always)]
    pub fn sub(a: Inner, b: Inner) -> Inner {
        zip(a, b, |x, y| x - y)
    }
    #[inline(always)]
    pub fn mul(a: Inner, b: Inner) -> Inner {
        zip(a, b, |x, y| x * y)
    }
    #[inline(always)]
    pub fn div(a: Inner, b: Inner) -> Inner {
        zip(a, b, |x, y| x / y)
    }
    #[inline(always)]
    pub fn mul_add(a: Inner, b: Inner, c: Inner) -> Inner {
        std::array::from_fn(|i| a[i].mul_add(b[i], c[i]))
    }
    /// The vector recipe: `x = n·ln2 + r`, a degree-5 polynomial for `e^r`,
    /// `2^n` through the exponent field.
    #[inline(always)]
    pub fn exp(x: Inner) -> Inner {
        each(x, |x| {
            let x = x.clamp(-88.0, 88.0);
            let n = (x * std::f32::consts::LOG2_E).round_ties_even();
            let r = n.mul_add(-0.693_359_4, x);
            let r = n.mul_add(2.121_944_4e-4, r);
            let mut p = r.mul_add(1.0 / 120.0, 1.0 / 24.0);
            p = r.mul_add(p, 1.0 / 6.0);
            p = r.mul_add(p, 0.5);
            p = r.mul_add(p, 1.0);
            p = r.mul_add(p, 1.0);
            p * f32::from_bits(((n as i32 + 127) as u32) << 23)
        })
    }
    #[inline(always)]
    pub fn sums8(v: [Inner; 8]) -> Inner {
        v.map(|l| ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7])))
    }
    #[inline(always)]
    pub fn lanes(v: Inner) -> [f32; 8] {
        v
    }
}

/// Eight `f32` lanes.
#[derive(Clone, Copy)]
pub struct V8(lanes::Inner);

impl V8 {
    #[inline(always)]
    pub fn splat(v: f32) -> V8 {
        V8(lanes::splat(v))
    }
    /// The first eight values of `s`.
    #[inline(always)]
    pub fn load(s: &[f32]) -> V8 {
        assert!(s.len() >= 8);
        // SAFETY: eight floats are in bounds.
        V8(unsafe { lanes::load(s.as_ptr()) })
    }
    /// Writes the lanes over the first eight values of `s`.
    #[inline(always)]
    pub fn store(self, s: &mut [f32]) {
        assert!(s.len() >= 8);
        // SAFETY: eight floats are in bounds.
        unsafe { lanes::store(s.as_mut_ptr(), self.0) }
    }
    /// # Safety
    /// `s[at..at + 8]` is in bounds.
    #[inline(always)]
    pub unsafe fn at(s: &[f32], at: usize) -> V8 {
        V8(unsafe { lanes::load(s.as_ptr().add(at)) })
    }
    /// # Safety
    /// `s[at..at + 8]` is in bounds.
    #[inline(always)]
    pub unsafe fn put(self, s: &mut [f32], at: usize) {
        unsafe { lanes::store(s.as_mut_ptr().add(at), self.0) }
    }
    #[inline(always)]
    pub fn mul_add(self, b: V8, c: V8) -> V8 {
        V8(lanes::mul_add(self.0, b.0, c.0))
    }
    #[inline(always)]
    pub fn exp(self) -> V8 {
        V8(lanes::exp(self.0))
    }
    /// `1 / (1 + e^-x)`.
    #[inline(always)]
    pub fn sigmoid(self) -> V8 {
        let one = V8::splat(1.0);
        one / (one + (-self).exp())
    }
    /// Lane `j` is the sum of `v[j]`'s lanes, folded
    /// `((0+1)+(2+3)) + ((4+5)+(6+7))`.
    #[inline(always)]
    pub fn sums8(v: [V8; 8]) -> V8 {
        V8(lanes::sums8(v.map(|x| x.0)))
    }
    /// The lanes' sum, folded `((0+4)+(2+6)) + ((1+5)+(3+7))`.
    #[inline(always)]
    pub fn sum(self) -> f32 {
        let l = lanes::lanes(self.0);
        ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]))
    }
}

impl std::ops::Add for V8 {
    type Output = V8;
    #[inline(always)]
    fn add(self, b: V8) -> V8 {
        V8(lanes::add(self.0, b.0))
    }
}

impl std::ops::Sub for V8 {
    type Output = V8;
    #[inline(always)]
    fn sub(self, b: V8) -> V8 {
        V8(lanes::sub(self.0, b.0))
    }
}

impl std::ops::Mul for V8 {
    type Output = V8;
    #[inline(always)]
    fn mul(self, b: V8) -> V8 {
        V8(lanes::mul(self.0, b.0))
    }
}

impl std::ops::Div for V8 {
    type Output = V8;
    #[inline(always)]
    fn div(self, b: V8) -> V8 {
        V8(lanes::div(self.0, b.0))
    }
}

impl std::ops::Neg for V8 {
    type Output = V8;
    #[inline(always)]
    fn neg(self) -> V8 {
        V8::splat(0.0) - self
    }
}

/// Runs `f` over eight-float blocks of `N` inputs and one output, the
/// shorter tail padded with zeros.
#[inline(always)]
pub fn map<const N: usize>(out: &mut [f32], inputs: [&[f32]; N], f: impl Fn([V8; N]) -> V8) {
    let n = out.len();
    for s in &inputs {
        assert!(s.len() >= n);
    }
    let whole = n - n % 8;
    let mut i = 0;
    while i < whole {
        // SAFETY: `i + 8 <= n`, and every input holds at least `n`.
        unsafe {
            let v = std::array::from_fn(|k| V8::at(inputs[k], i));
            f(v).put(out, i);
        }
        i += 8;
    }
    if i < n {
        let v = std::array::from_fn(|k| {
            let mut pad = [0f32; 8];
            pad[..n - i].copy_from_slice(&inputs[k][i..n]);
            V8::load(&pad)
        });
        let mut pad = [0f32; 8];
        f(v).store(&mut pad);
        out[i..].copy_from_slice(&pad[..n - i]);
    }
}

/// [`map`] with the output also an input, as its first argument.
#[inline(always)]
pub fn update<const N: usize>(out: &mut [f32], inputs: [&[f32]; N], f: impl Fn(V8, [V8; N]) -> V8) {
    let n = out.len();
    for s in &inputs {
        assert!(s.len() >= n);
    }
    let whole = n - n % 8;
    let mut i = 0;
    while i < whole {
        // SAFETY: `i + 8 <= n`, and every input holds at least `n`.
        unsafe {
            let v = std::array::from_fn(|k| V8::at(inputs[k], i));
            f(V8::at(out, i), v).put(out, i);
        }
        i += 8;
    }
    if i < n {
        let v = std::array::from_fn(|k| {
            let mut pad = [0f32; 8];
            pad[..n - i].copy_from_slice(&inputs[k][i..n]);
            V8::load(&pad)
        });
        let mut pad = [0f32; 8];
        pad[..n - i].copy_from_slice(&out[i..]);
        f(V8::load(&pad), v).store(&mut pad);
        out[i..].copy_from_slice(&pad[..n - i]);
    }
}

/// The sum over the elements of `f` of `N` equally long inputs: four
/// accumulators over 32-float blocks, then one over 8, the tail padded with
/// zeros, folded `(a0 + a1) + (a2 + a3)` and then across the lanes.
#[inline(always)]
pub fn sum<const N: usize>(inputs: [&[f32]; N], f: impl Fn([V8; N]) -> V8) -> f32 {
    let n = inputs[0].len();
    for s in &inputs {
        assert!(s.len() >= n);
    }
    let zero = V8::splat(0.0);
    let mut acc = [zero; 4];
    let mut i = 0;
    while i + 32 <= n {
        for (a, slot) in acc.iter_mut().enumerate() {
            let at = i + 8 * a;
            // SAFETY: `at + 8 <= n`, and every input holds at least `n`.
            *slot = *slot + f(std::array::from_fn(|k| unsafe { V8::at(inputs[k], at) }));
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: as above.
        acc[0] = acc[0] + f(std::array::from_fn(|k| unsafe { V8::at(inputs[k], i) }));
        i += 8;
    }
    if i < n {
        let v = std::array::from_fn(|k| {
            let mut pad = [0f32; 8];
            pad[..n - i].copy_from_slice(&inputs[k][i..n]);
            V8::load(&pad)
        });
        // The padded lanes are zero in every input; `f` of zeros is not
        // necessarily zero, so the lanes past the end are masked.
        let mut mask = [0f32; 8];
        mask[..n - i].fill(1.0);
        acc[0] = acc[0] + f(v) * V8::load(&mask);
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * k).sin() * 3.0).collect()
    }

    /// The lanes' `exp` is the library's to a few parts in a million over
    /// the range a gate sees, and saturates rather than overflowing.
    #[test]
    fn exp_is_the_librarys_to_rounding() {
        let x = wave(1003, 0.37);
        let mut got = vec![0f32; x.len()];
        map(&mut got, [&x], |[v]| v.exp());
        for (g, v) in got.iter().zip(&x) {
            let want = v.exp();
            assert!((g - want).abs() <= 4e-6 * want, "{v}: {g} vs {want}");
        }
        let mut big = [0f32; 1];
        map(&mut big, [&[1000.0]], |[v]| v.exp());
        assert!(big[0].is_finite());
    }

    /// Every length, tails included: elementwise kernels match the scalar
    /// loop, and sums match it to rounding.
    #[test]
    fn kernels_match_the_plain_loop_at_every_length() {
        for n in [0, 1, 7, 8, 9, 31, 32, 33, 100, 768] {
            let a = wave(n, 0.11);
            let b = wave(n, 0.53);
            let mut got = vec![0f32; n];
            map(&mut got, [&a, &b], |[x, y]| x * y + x);
            for i in 0..n {
                assert_eq!(got[i], a[i] * b[i] + a[i]);
            }
            let mut acc = b.clone();
            update(&mut acc, [&a], |o, [x]| o + x);
            for i in 0..n {
                assert_eq!(acc[i], b[i] + a[i]);
            }
            let s = sum([&a, &b], |[x, y]| x * y + V8::splat(1.0));
            let want: f64 = (0..n)
                .map(|i| f64::from(a[i]) * f64::from(b[i]) + 1.0)
                .sum();
            assert!(
                (f64::from(s) - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "{n}"
            );
        }
    }

    /// Eight vectors' sums, in the documented fold, on every machine.
    #[test]
    fn sums8_folds_in_one_order() {
        let x = wave(64, 0.77);
        let v: [V8; 8] = std::array::from_fn(|i| V8::load(&x[i * 8..]));
        let mut got = [0f32; 8];
        V8::sums8(v).store(&mut got);
        for (j, g) in got.iter().enumerate() {
            let l = &x[j * 8..j * 8 + 8];
            let want = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
            assert_eq!(g.to_bits(), want.to_bits());
        }
    }

    /// A sigmoid stays in (0, 1] at both ends of the range.
    #[test]
    fn sigmoid_saturates() {
        let x = [-200.0, -20.0, 0.0, 20.0, 200.0, 1.0, -1.0, 3.0];
        let mut got = [0f32; 8];
        map(&mut got, [&x], |[v]| v.sigmoid());
        assert_eq!(got[2], 0.5);
        assert!(got[0] >= 0.0 && got[0] < 1e-30);
        assert_eq!(got[4], 1.0);
        for (g, v) in got.iter().zip(&x) {
            assert!((g - 1.0 / (1.0 + (-v).exp())).abs() < 1e-6);
        }
    }
}
