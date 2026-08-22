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

use super::*;

/// Every spelling the config file and the override accept, and the two
/// that are deliberately the same value.
#[test]
fn kv_cache_names_round_trip() {
    use crate::config::KvCache;
    for value in [KvCache::F16, KvCache::Q8_0, KvCache::F32] {
        assert_eq!(KvCache::parse(value.tag()), Some(value), "{}", value.tag());
    }
    // Case and padding are the shapes a hand-edited config file produces.
    assert_eq!(KvCache::parse("  Q8_0 "), Some(KvCache::Q8_0));
    assert_eq!(KvCache::parse("q8"), Some(KvCache::Q8_0));
    // Anything else is an error at the call site, never a silent default:
    // a typo that fell back to `f16` would be a quality setting nobody
    // chose and nothing would say so.
    assert_eq!(KvCache::parse("int8"), None);
    assert_eq!(KvCache::parse(""), None);
}

/// The default is `f16`, and a process that never set a preference gets
/// it — which is every test, and every backend built without a config.
#[test]
fn the_unset_preference_is_f16() {
    assert_eq!(kv_cache_preference(), crate::config::KvCache::F16);
}
