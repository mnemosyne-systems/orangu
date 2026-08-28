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

//! A `f32` buffer that starts on a cache line.
//!
//! `Vec<f32>` promises four-byte alignment and the allocator happens to
//! give sixteen. That is the wrong number for eight-wide loads: a 32-byte
//! load from a 16-byte-aligned base sits at offset 16 within its cache
//! line half the time, and those loads *cross* into the next line. The
//! hardware handles it, at the cost of a second cache access.
//!
//! Measured on the smoke model before this existed: **28.8 billion
//! misaligned accesses out of 101 billion loads, 28.5%**. Every one of
//! them is a vector load of a weight or an activation, and none of them
//! had to be misaligned — the offsets *within* the buffer are all
//! multiples of eight floats already, because every tensor's row length is.
//! Only the base was wrong.
//!
//! So this is the whole of the data structure: a `Vec` of cache lines,
//! reinterpreted as floats. The alignment comes from the element type
//! rather than from an allocator flag, which is what keeps it safe code,
//! and the length is rounded up to a whole line so the tail of the last
//! vector load is inside the allocation.

use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

// One cache line of floats. `align(64)` is what the `Vec` inherits.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Line([f32; Line::FLOATS]);

impl Line {
    const FLOATS: usize = 16;
}

// Buffers a thread keeps to hand out again.
//
// A training step allocates and frees roughly two hundred of these — a
// dozen per block forward, again for the recompute, again for the
// backward. Freeing them returns the pages to the kernel, and asking for
// them again takes them back one page fault at a time: **575,212 minor
// faults over a twenty-step run**, each one a trap into the kernel and a
// page of zeroing that the allocation is about to overwrite anyway.
//
// Recycling them keeps the pages mapped. The buffer is still zeroed on
// the way out, because callers are entitled to that, but zeroing memory
// that is already resident is a `memset` rather than a fault storm.
//
// The pool is thread-local, which is what makes it lock-free and what
// makes it work under a work-stealing pool: a buffer allocated on one
// thread and dropped on another simply joins the second thread's list.
// It is bounded twice — by count and by total bytes — so a run that
// briefly needs a large buffer does not keep it forever.
thread_local! {
    static POOL: RefCell<Vec<Vec<Line>>> = const { RefCell::new(Vec::new()) };
}

// Buffers one thread keeps, and the total they may occupy. Sized for a
// block's working set rather than a whole model: what the pool is for is
// the churn inside a step, not holding the model twice.
const POOL_COUNT: usize = 48;
const POOL_BYTES: usize = 128 << 20;

// Takes a buffer of at least `lines` lines from this thread's pool.
//
// Best fit, not first fit: handing a 32 MiB buffer to a request for 8 KiB
// would keep the large one busy and force the next large request to
// allocate.
fn take(lines: usize) -> Option<Vec<Line>> {
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let best = pool
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.capacity() >= lines)
            .min_by_key(|(_, buffer)| buffer.capacity())
            .map(|(at, _)| at)?;
        Some(pool.swap_remove(best))
    })
}

// Returns a buffer to this thread's pool, or drops it if the pool is full.
fn give(buffer: Vec<Line>) {
    if buffer.capacity() == 0 {
        return;
    }
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let held: usize = pool.iter().map(|b| b.capacity() * size_of::<Line>()).sum();
        if pool.len() < POOL_COUNT && held + buffer.capacity() * size_of::<Line>() <= POOL_BYTES {
            pool.push(buffer);
        }
    });
}

// A `f32` buffer whose start is 64-byte aligned.
//
// Derefs to `[f32]`, so it is used exactly like the `Vec<f32>` it
// replaces; the padding past `len` is never visible through that slice.
#[derive(Clone)]
pub struct Aligned {
    lines: Vec<Line>,
    len: usize,
}

impl Aligned {
    pub fn zeros(len: usize) -> Self {
        let lines = len.div_ceil(Line::FLOATS);
        let mut buffer = take(lines).unwrap_or_else(|| Vec::with_capacity(lines));
        buffer.clear();
        buffer.resize(lines, Line([0.0; Line::FLOATS]));
        Aligned { lines: buffer, len }
    }

    // Copies an existing buffer into an aligned one — the checkpoint
    // reader's path, which has no say in how its bytes arrived.
    pub fn from_slice(values: &[f32]) -> Self {
        let mut out = Aligned::zeros(values.len());
        out.copy_from_slice(values);
        out
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        give(std::mem::take(&mut self.lines));
    }
}

impl Default for Aligned {
    fn default() -> Self {
        Aligned::zeros(0)
    }
}

impl Deref for Aligned {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &bytemuck::cast_slice(&self.lines)[..self.len]
    }
}

impl DerefMut for Aligned {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut bytemuck::cast_slice_mut(&mut self.lines)[..self.len]
    }
}

impl std::fmt::Debug for Aligned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Aligned({} floats)", self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The one thing this type exists for.
    #[test]
    fn the_buffer_starts_on_a_cache_line() {
        for len in [1usize, 7, 8, 16, 17, 1000, 5_262_080] {
            let buffer = Aligned::zeros(len);
            assert_eq!(buffer.len(), len);
            assert_eq!(
                buffer.as_ptr() as usize % 64,
                0,
                "a {len}-float buffer did not start on a cache line"
            );
            assert!(buffer.iter().all(|&v| v == 0.0));
        }
    }

    // Every eight-float load inside it is aligned too, which is the
    // property the kernels actually depend on: tensor offsets are all
    // multiples of eight, so a 64-byte-aligned base makes every one of
    // them 32-byte aligned.
    #[test]
    fn every_eight_float_offset_is_vector_aligned() {
        let buffer = Aligned::zeros(4096);
        for offset in (0..4096).step_by(8) {
            let address = buffer[offset..].as_ptr() as usize;
            assert_eq!(address % 32, 0, "offset {offset} is not vector-aligned");
        }
    }

    // A recycled buffer must arrive zeroed, or a caller that only writes
    // part of it inherits the last tensor that lived there.
    #[test]
    fn a_recycled_buffer_comes_back_zeroed() {
        let mut first = Aligned::zeros(1024);
        first.fill(7.0);
        let address = first.as_ptr() as usize;
        drop(first);

        let second = Aligned::zeros(1024);
        assert!(
            second.iter().all(|&v| v == 0.0),
            "a reused buffer kept its old contents"
        );
        assert_eq!(
            second.as_ptr() as usize,
            address,
            "the buffer should have been recycled, not reallocated"
        );
        assert_eq!(second.as_ptr() as usize % 64, 0, "still cache-line aligned");
    }

    // The pool is bounded, so a run that briefly needs something large
    // does not hold onto it.
    #[test]
    fn the_pool_is_bounded() {
        // Far more buffers than the pool keeps.
        let held: Vec<Aligned> = (0..POOL_COUNT * 2).map(|_| Aligned::zeros(64)).collect();
        drop(held);
        let kept = POOL.with(|pool| pool.borrow().len());
        assert!(kept <= POOL_COUNT, "the pool kept {kept} buffers");

        // And a buffer past the byte budget is dropped rather than kept.
        drop(Aligned::zeros(POOL_BYTES));
        let bytes: usize = POOL.with(|pool| {
            pool.borrow()
                .iter()
                .map(|b| b.capacity() * size_of::<Line>())
                .sum()
        });
        assert!(bytes <= POOL_BYTES, "the pool holds {bytes} bytes");
    }

    #[test]
    fn it_behaves_like_the_vector_it_replaces() {
        let mut buffer = Aligned::zeros(10);
        buffer[3] = 1.5;
        buffer.fill(2.0);
        assert_eq!(buffer.len(), 10);
        assert!(buffer.iter().all(|&v| v == 2.0));

        let copied = Aligned::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(&copied[..], &[1.0, 2.0, 3.0]);
    }

    // The padding to a whole line must not be visible.
    #[test]
    fn the_padding_is_not_part_of_the_buffer() {
        let buffer = Aligned::zeros(17);
        assert_eq!(buffer.len(), 17);
        assert_eq!(buffer.iter().count(), 17);
    }
}
