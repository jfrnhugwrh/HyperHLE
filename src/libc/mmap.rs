/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use crate::dyld::FunctionExports;
use crate::environment::Environment;
use crate::export_c_func;
use crate::libc::errno::{set_errno, EINVAL, EIO, ENOMEM};
use crate::libc::posix_io;
use crate::libc::posix_io::{off_t, open_direct, FileDescriptor, SEEK_SET};
use crate::mem::{ConstPtr, GuestUSize, MutVoidPtr, PAGE_SIZE, PAGE_SIZE_ALIGN_MASK};
use std::collections::HashMap;

#[allow(dead_code)]
const MAP_FILE: i32 = 0x0000;
const MAP_ANON: i32 = 0x1000;
#[allow(dead_code)]
const MAP_FIXED: i32 = 0x0010;
const MAP_FAILED: MutVoidPtr = MutVoidPtr::from_bits(0xffff_ffff);

#[derive(Default)]
pub struct State {
    /// Keeping track of `mmap` allocations
    allocations: HashMap<MutVoidPtr, GuestUSize>,
}

/// For files, our implementation of mmap is really simple:
/// it's just load entirety of file in memory!
fn mmap(
    env: &mut Environment,
    addr: MutVoidPtr,
    len: GuestUSize,
    prot: i32,
    flags: i32,
    fd: FileDescriptor,
    offset: off_t,
) -> MutVoidPtr {
    // TODO: handle errno properly
    set_errno(env, 0);
    log_dbg!(
        "mmap({:?}, {}, {}, {}, {}, {})",
        addr,
        len,
        prot,
        flags,
        fd,
        offset
    );

    // FIXME: Experimental Hack (HyperHLE compatibility)
    // Target: Puzzle Agent 1.2 (com.telltalegames.Grickle101Low), Telltale Tool
    // Purpose: make guest mmap() behave like Darwin's page-granular mapper
    //          instead of panicking or returning bogus pointers.
    // Assumptions:
    //   1. Darwin always returns page-aligned, page-granular mappings, so we
    //      round the allocation up to a full page. Our guest heap only
    //      guarantees MIN_CHUNK_SIZE (16-byte) alignment for sub-page
    //      requests, which used to trip the page-alignment assert below for
    //      any MAP_ANON mapping smaller than one page.
    //   2. The address hint (even with MAP_FIXED) can be ignored: Puzzle
    //      Agent, like most iOS games, uses mmap for anonymous scratch
    //      buffers and file-backed loads and does not require an exact
    //      placement. Verify by continuing execution beyond mmap.
    // If the game ever needs true fixed-address mappings, MAP_FIXED handling
    // must be implemented in the guest allocator instead.
    if (flags & MAP_FIXED) != 0 && !addr.is_null() {
        log_dbg!(
            "Warning: mmap: MAP_FIXED requested for {:?}; ignoring hint (fixed mappings unsupported)",
            addr
        );
    }

    // POSIX: mmap() with a zero length fails with EINVAL and MAP_FAILED.
    if len == 0 {
        log!("Warning: mmap: zero length; returning MAP_FAILED (EINVAL)");
        set_errno(env, EINVAL);
        return MAP_FAILED;
    }

    // Round the request up to page granularity (checked — a request near
    // 0xFFFFFFFF overflows here and must fail instead of wrapping).
    let alloc_len = match len.checked_next_multiple_of(PAGE_SIZE) {
        Some(alloc_len) => alloc_len,
        None => {
            log!(
                "Warning: mmap: length {:#x} overflows when page-rounded; returning MAP_FAILED (ENOMEM)",
                len
            );
            set_errno(env, ENOMEM);
            return MAP_FAILED;
        }
    };

    // TODO: use vm_allocate() instead
    let ptr = env.mem.calloc(alloc_len);

    // The guest allocator can return NULL (out of memory, or a size that
    // overflows after alignment). Real kernels never return NULL from mmap()
    // — failure is always MAP_FAILED with errno set — so a NULL here must be
    // translated, otherwise the guest treats address 0 as a valid mapping and
    // faults on its first dereference.
    if ptr.is_null() {
        log!(
            "Warning: mmap({:?}, {}) failed: guest allocator returned NULL; returning MAP_FAILED (ENOMEM)",
            addr, len
        );
        set_errno(env, ENOMEM);
        return MAP_FAILED;
    }

    if (flags & MAP_ANON) != 0 {
        // Guaranteed by the page-rounded allocation above.
        assert!(ptr.to_bits() & PAGE_SIZE_ALIGN_MASK == 0);

        // Darwin ignores `fd` and `offset` entirely when MAP_ANON is set.
        // Engines (e.g. Adobe AIR) pass garbage there.
        if fd != -1 || offset != 0 {
            log_dbg!(
                "Warning: mmap MAP_ANON called with fd={} and offset={}. Ignoring them as per OS behavior.",
                fd,
                offset
            );
        }

        if !addr.is_null() {
            // POSIX `mmap` documents the `addr` argument as a hint that
            // implementations may ignore. We always allocate from the
            // guest heap, so the actual placement is the heap allocator's
            // choice. Apps that genuinely require fixed-address mappings
            // would set MAP_FIXED, which we'd then need to honour
            // separately. Demoted to debug to keep Mono/Unity startup
            // logs readable; the host-vs-hint mismatch is not an error.
            log_dbg!(
                "mmap MAP_ANON ignoring hint for address {:?}, actual is {:?}",
                addr,
                ptr
            );
        }
    } else {
        // File-backed mmap: read file content into the allocated buffer.
        if !addr.is_null() {
            log_dbg!(
                "mmap file-backed ignoring hint for address {:?}, actual is {:?}",
                addr,
                ptr
            );
        }
        // Seek to the requested offset. If the seek fails (e.g. bad fd),
        // return MAP_FAILED (-1 as pointer) instead of crashing.
        let new_offset = posix_io::lseek(env, fd, offset, SEEK_SET);
        if new_offset != offset {
            log!(
                "Warning: mmap: lseek to offset {} failed (returned {}); returning MAP_FAILED",
                offset,
                new_offset
            );
            env.mem.free(ptr);
            set_errno(env, EIO);
            return MAP_FAILED;
        }

        let read = posix_io::read(env, fd, ptr, len);
        if read < 0 {
            log!(
                "Warning: mmap: read of {} bytes from fd {} failed (errno path); buffer left zero-filled",
                len, fd
            );
        } else if (read as u32) < len {
            log!(
                "Warning: mmap: read only {} of {} bytes from fd {}; padding remainder with zeros",
                read,
                len,
                fd
            );
            // Remainder is already zeroed (calloc)
        }
    }

    assert!(!env.libc_state.mmap.allocations.contains_key(&ptr));
    env.libc_state.mmap.allocations.insert(ptr, len);

    ptr
}

fn munmap(env: &mut Environment, addr: MutVoidPtr, len: GuestUSize) -> i32 {
    // TODO: handle errno properly
    set_errno(env, 0);
    log_dbg!("munmap({:?}, {})", addr, len);

    if len == 0 {
        set_errno(env, EINVAL);
        // TODO: should we clear allocations for `addr` here too?
        log!("Warning: munmap({:?}, {}) failed, returning -1", addr, len);
        return -1;
    }

    if let Some(&expected_len) = env.libc_state.mmap.allocations.get(&addr) {
        if expected_len != len {
            log_dbg!(
                "munmap({:?}, {}): length mismatch (expected {}), proceeding anyway",
                addr,
                len,
                expected_len
            );
        }
        env.mem.free(addr);
        env.libc_state.mmap.allocations.remove(&addr);
        0 // success
    } else {
        // FIXME: Experimental Hack (HyperHLE compatibility)
        // Target: Puzzle Agent 1.2 (com.telltalegames.Grickle101Low)
        // Purpose: untracked munmap() calls must not fail the game.
        // Assumption: a region we didn't hand out via mmap() (e.g. obtained
        // via vm_allocate, malloc, or double-unmapped) can be silently left
        // alone, matching the lenient behaviour guest engines expect.
        // Darwin only fails munmap for fully-unmapped ranges; failing here
        // makes engines treat teardown as fatal. Verify by continuing
        // execution beyond munmap.
        if addr.is_null() {
            set_errno(env, EINVAL);
            log!(
                "Warning: munmap({:?}, {}) failed, returning -1 (NULL address)",
                addr,
                len
            );
            -1
        } else {
            log_dbg!(
                "munmap({:?}, {}): unknown mapping, succeeding as no-op (compatibility)",
                addr,
                len
            );
            set_errno(env, 0);
            0
        }
    }
}

fn madvise(env: &mut Environment, addr: MutVoidPtr, len: GuestUSize, advice: i32) -> i32 {
    // FIXME: Experimental Hack (HyperHLE compatibility)
    // Target: Puzzle Agent 1.2 (com.telltalegames.Grickle101Low), Telltale Tool
    // Purpose: madvise() must succeed so engine buffer management proceeds.
    // Assumption: touchHLE keeps all guest memory resident and readable/
    // writable at all times, so any MADV_* advice is trivially already
    // satisfied. Real Darwin returns 0 for valid advice; returning -1 +
    // ENOTSUP makes engines treat paging setup as failed and abort.
    // Verify by continuing execution beyond madvise.
    log_dbg!(
        "madvise({:?}, {}, {}) -> 0 (no-op; guest memory is always resident)",
        addr,
        len,
        advice
    );
    set_errno(env, 0);
    0
}

fn shm_open(env: &mut Environment, name: ConstPtr<u8>, oflag: i32, mode: u32) -> i32 {
    set_errno(env, 0);

    let name_str = env.mem.cstr_at_utf8(name).unwrap_or("<invalid>");
    log_dbg!("shm_open({:?}, {:#x}, {:#x})", name_str, oflag, mode);

    // Используем open_direct! Параметр mode для эмулятора здесь не нужен,
    // поэтому просто передаем env, name и oflag.
    open_direct(env, name, oflag)
}

fn mprotect(env: &mut Environment, addr: MutVoidPtr, len: GuestUSize, prot: i32) -> i32 {
    // POSIX `int mprotect(void *addr, size_t len, int prot)`: returns 0
    // on success, -1 on failure with errno set.
    //
    // touchHLE doesn't enforce per-page memory protections — the entire
    // guest address space is treated as RW (and code pages as RX through
    // the JIT). However, returning -1 + ENOTSUP for every mprotect call
    // is wrong: it makes Mono/Boehm GC and Unity's runtime think the
    // protection change failed during JIT/GC initialization, which can
    // leave them in a broken state. Real Darwin kernels never fail
    // mprotect for the address ranges that mmap'd allocations sit in,
    // so the correct behavior is "succeed silently as a no-op".
    //
    // Reference: POSIX mprotect(2) — return value is 0 on success, -1 on
    // error; the only documented errors apply to invalid/non-mapped
    // address ranges, which our guest mmap allocator handles by always
    // returning a valid range.
    log_dbg!(
        "mprotect({:?}, {}, {:#x}) -> 0 (no-op; touchHLE does not enforce per-page protections)",
        addr,
        len,
        prot
    );
    set_errno(env, 0);
    0
}

/// `int mlock(const void *addr, size_t len)` — lock a region of memory
/// so it stays resident in physical RAM. touchHLE keeps the entire
/// guest address space resident in the host process at all times, so
/// there is nothing to pin: the memory is already non-pageable from the
/// guest's perspective. Real Darwin returns 0 on success, so we report
/// success as a no-op (returning -1 here would make guest code that
/// relies on mlock — e.g. crypto/keychain libraries protecting secrets,
/// or audio engines pinning buffers — treat initialization as failed).
///
/// Reference: POSIX/Darwin mlock(2) — returns 0 on success, -1 with
/// errno on failure.
fn mlock(env: &mut Environment, addr: ConstPtr<u8>, len: GuestUSize) -> i32 {
    log_dbg!(
        "mlock({:?}, {}) -> 0 (no-op; guest memory is always resident)",
        addr,
        len
    );
    set_errno(env, 0);
    0
}

/// `int munlock(const void *addr, size_t len)` — unlock a region
/// previously locked with `mlock`. Mirrors [mlock]: a no-op that
/// succeeds.
///
/// Reference: POSIX/Darwin munlock(2) — returns 0 on success.
fn munlock(env: &mut Environment, addr: ConstPtr<u8>, len: GuestUSize) -> i32 {
    log_dbg!(
        "munlock({:?}, {}) -> 0 (no-op; guest memory is always resident)",
        addr,
        len
    );
    set_errno(env, 0);
    0
}

pub const FUNCTIONS: FunctionExports = &[
    export_c_func!(mmap(_, _, _, _, _, _)),
    export_c_func!(munmap(_, _)),
    export_c_func!(madvise(_, _, _)),
    export_c_func!(shm_open(_, _, _)),
    export_c_func!(mprotect(_, _, _)),
    export_c_func!(mlock(_, _)),
    export_c_func!(munlock(_, _)),
];
