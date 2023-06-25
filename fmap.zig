const std = @import("std");
const testing = std.testing;
const print = std.debug.print;

const CAPACITY = 14;
const DESIRED_CAP = CAPACITY - 2;
const FULL_MASK = ((1 << CAPACITY) - 1);

fn last_set_idx_nonzero(mask: u32) u32 {
    return (1 + (31 ^ @clz(mask)));
}

fn find_last_set(val: u32) u32 {
    return if (val == 0) 0 else last_set_idx_nonzero(val);
}

const fmap_chunk_head = extern struct {
    tags: [14]u8,
    control: i8,
    overflow: u8,
    const Self = @This();

    fn new() Self {
        return std.mem.zeroInit(Self, .{});
    }

    fn clear(h: *Self) void {
        h.* = std.mem.zeroInit(Self, .{});
    }

    fn mark_eof(h: *Self, scale: u32) void {
        h.control = ((h.control & ~@intCast(i8, 0x0f) | @intCast(i8, scale)));
    }

    fn get_scale(h: Self) u32 {
        return @intCast(u32, (h.control & 0x0f));
    }

    fn idx_used(h: Self, idx: usize) bool {
        return h.tags[idx] != 0;
    }

    fn set_tag(h: *Self, idx: usize, tag: u32) void {
        h.tags[idx] = @intCast(u8, tag);
    }

    fn clear_tag(h: *Self, idx: usize) void {
        h.tags[idx] = @intCast(u8, 0);
    }

    fn hosted_overflow_count(h: Self) u32 {
        return @intCast(u32, h.control >> 4);
    }

    fn adj_hosted_overflow_count(h: *Self, hostedOp: i8) void {
        h.control += hostedOp;
    }

    fn inc_overflow_count(h: *Self) void {
        if (h.overflow != 255)
            h.overflow += 1;
    }

    fn dec_overflow_count(h: *Self) void {
        if (h.overflow != 255)
            h.overflow -= 1;
    }

    fn overflow_count(h: Self) u8 {
        return h.overflow;
    }

    fn get_tag(h: Self, idx: usize) u32 {
        return @intCast(u32, h.tags[idx]);
    }

    fn tag_vector(h: Self) @Vector(16, u8) {
        return @bitCast(@Vector(16, u8), h);
    }

    fn occupied_mask(h: Self) u16 {
        const v = h.tag_vector();
        const mask = @bitCast(u16, (v != @splat(16, @intCast(u8, 0))));
        return mask & FULL_MASK;
    }

    fn is_eof(h: Self) bool {
        return (h.control & 0xF) != 0;
    }

    fn last_occupied_idx(h: Self) ?u16 {
        var mask = h.occupied_mask();
        if (mask == 0)
            return null;
        return @intCast(u16, last_set_idx_nonzero(mask)) - 1;
    }

    fn first_empty_idx(h: Self) ?u16 {
        var mask = h.occupied_mask() ^ FULL_MASK;
        if (mask == 0)
            return null;
        return @ctz(mask);
    }
};

fn fmap_chunk(comptime item_type: type) type {
    return extern struct {
        head: fmap_chunk_head align(16),
        items: [14]item_type,

        const Self = @This();

        fn new() Self {
            return std.mem.zeroInit(Self, .{});
        }
    };
}

test {
    var h = fmap_chunk_head.new();
    try testing.expect(@TypeOf(h) == fmap_chunk_head);
    var v0 = h.tag_vector();
    var v1 = @splat(16, @intCast(u8, 0));
    try testing.expect(@reduce(.And, (v0 == v1)) == true);
}

test {
    var h = fmap_chunk_head.new();
    h.set_tag(0, 1);
    try testing.expect(h.occupied_mask() == 0x1);
    h.set_tag(1, 2);
    try testing.expect(h.occupied_mask() == 0x3);
}

test {
    var h = fmap_chunk_head.new();
    h.set_tag(0, 1);
    try testing.expect(h.last_occupied_idx().? == 0);
    h.set_tag(1, 2);
    try testing.expect(h.last_occupied_idx().? == 1);
}

const dense_iter = struct {
    mask: u16,
    index: u16,
    const Self = @This();

    fn new(mask: u16) Self {
        return .{ .mask = mask, .index = 0 };
    }

    fn has_next(iter: Self) bool {
        return iter.mask != 0;
    }

    fn from_chunk_head(h: fmap_chunk_head) Self {
        return new(h.occupied_mask());
    }

    fn next(iter: *Self) ?u16 {
        if ((iter.mask & 1) != 0) {
            iter.mask >>= 1;
            return blk: {
                const index = iter.index;
                iter.index += 1;
                break :blk index;
            };
        } else {
            if (iter.mask == 0)
                return null;

            const s = @ctz(iter.mask);
            const idx = iter.index + s;
            const mask: u32 = iter.mask;
            iter.mask = @intCast(u16, mask >> (s + 1));
            iter.index = @intCast(u16, idx + 1);
            return idx;
        }
    }
};

test {
    var h = fmap_chunk_head.new();
    h.set_tag(0, 1);
    h.set_tag(1, 2);
    h.set_tag(2, 0);
    h.set_tag(3, 1);

    var iter = dense_iter.from_chunk_head(h);

    try testing.expect(iter.has_next() == true);

    var expect_idx: u16 = 0;
    const expects = [_]u16{ 0, 1, 3 };

    while (iter.next()) |idx| {
        try testing.expect(idx == expects[expect_idx]);
        expect_idx += 1;
    }
}

test {
    var h = fmap_chunk_head.new();
    for (0..10) |idx| {
        h.set_tag(idx, 1);
    }
    h.set_tag(11, 1);

    var iter = dense_iter.from_chunk_head(h);
    try testing.expect(iter.has_next() == true);

    var expect_idx: u16 = 0;
    const expects = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11 };

    while (iter.next()) |idx| {
        try testing.expect(idx == expects[expect_idx]);
        expect_idx += 1;
    }
}

const match_iter = struct {
    mask: u16,
    const self = @This();

    fn new(h: fmap_chunk_head, needle: u8) self {
        const v = h.tag_vector();
        const needle_v = @splat(16, needle);
        const result = @bitCast(u16, v == needle_v);
        return .{ .mask = result };
    }

    fn has_next(iter: self) bool {
        return iter.mask != 0;
    }

    fn next(iter: *self) ?u16 {
        if (!iter.has_next())
            return null;

        var idx = @ctz(iter.mask);
        iter.mask &= (iter.mask - 1);
        return idx;
    }
};

test {
    var h = fmap_chunk_head.new();
    for (0..10) |idx| {
        h.set_tag(idx, 1);
    }
    h.set_tag(11, 1);

    var iter = match_iter.new(h, 1);
    try testing.expect(iter.has_next() == true);

    var expect_idx: u16 = 0;
    const expects = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11 };

    while (iter.next()) |idx| {
        try testing.expect(idx == expects[expect_idx]);
        expect_idx += 1;
    }
}

test {
    var h = fmap_chunk_head.new();
    h.set_tag(0, 1);
    h.set_tag(1, 2);
    h.set_tag(2, 0);
    h.set_tag(3, 1);

    var iter = match_iter.new(h, 1);
    try testing.expect(iter.has_next() == true);

    var expect_idx: u16 = 0;
    const expects = [_]u16{ 0, 3 };

    while (iter.next()) |idx| {
        try testing.expect(idx == expects[expect_idx]);
        expect_idx += 1;
    }
}

fn item_iter(comptime item_type: type) type {
    const chunk_type = fmap_chunk(item_type);
    return struct {
        item: ?[*]item_type,
        index: usize,

        const iter_type = @This();

        fn new(chunk: *chunk_type, index: usize) iter_type {
            return .{ .item = @ptrCast([*]item_type, &chunk.items[index]), .index = index };
        }

        fn to_chunk(iter: iter_type) [*]chunk_type {
            var item = iter.item.?;
            item -= iter.index;
            const addr = @fieldParentPtr(chunk_type, "items", @ptrCast(*[14]item_type, item));
            return @ptrCast([*]chunk_type, addr);
        }

        fn at_end(iter: iter_type) bool {
            return iter.item == null;
        }

        inline fn advanceImpl(self: *iter_type, comptime check_oef: bool, comptime likely_dead: bool) void {
            var chunk = self.to_chunk();
            var h = &chunk[0].head;
            while (self.index > 0) {
                self.index -= 1;
                self.item = self.item.? - 1;
                if (h.idx_used(self.index))
                    return;
            }

            while (true) {
                if (check_oef) {
                    if (h.is_eof()) {
                        self.item = null;
                        return;
                    }
                }
                chunk -= 1;
                h = &chunk[0].head;

                if (check_oef and !likely_dead) {
                    @prefetch(chunk - 1, .{});
                }

                if (h.last_occupied_idx()) |idx| {
                    self.item = @ptrCast([*]item_type, &chunk[0].items[idx]);
                    self.index = idx;
                    return;
                }
            }
        }

        fn advance_likely_dead(self: *iter_type) void {
            return advanceImpl(self, true, true);
        }

        fn advance_prechecked(self: *iter_type) void {
            return advanceImpl(self, false, false);
        }
    };
}

test {
    const chunk_type = fmap_chunk(u64);
    var chunk = chunk_type.new();
    var iter = item_iter(u64).new(&chunk, 2);
    try testing.expect(@ptrCast([*]chunk_type, &chunk) == iter.to_chunk());
}

test {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var chunk = try allocator.alloc(fmap_chunk(u64), 100);
    @memset(std.mem.sliceAsBytes(chunk), 0);

    defer allocator.free(chunk);

    chunk[0].head.mark_eof(1);

    for (0..20) |idx| {
        var h = &(chunk[idx].head);
        h.set_tag(0, 1);
        h.set_tag(1, 2);
        h.set_tag(3, 2);
    }
    var iter = item_iter(u64).new(&chunk[19], 0);

    var idx: usize = 0;
    while (!iter.at_end()) {
        iter.advanceImpl(true, false);
        idx += 1;
    }

    try testing.expect(idx == 58);
}

const hash_pair = struct {
    hash: u32,
    tag: u32,

    fn probe_delta(self: @This()) u32 {
        return 2 * self.tag + 1;
    }

    fn from_hash(hash: u32) @This() {
        return .{ .hash = hash, .tag = ((hash >> 24) | 0x80) };
    }
};

const Allocator = std.mem.Allocator;

fn fmap(comptime item_type: type, comptime hasher: type) type {
    const chunk_type = fmap_chunk(item_type);
    const chunk_ptr_type = [*]chunk_type;
    const item_iter_type = item_iter(item_type);
    const S = struct {
        var empty_chunk = std.mem.zeroes(chunk_type);
    };

    return struct {
        chunk_ptr: chunk_ptr_type,
        chunk_mask: u32,
        size: u32,
        flags: u32,

        const Self = @This();

        fn is_empty(self: *Self) bool {
            return self.chunk_ptr == @ptrCast(chunk_ptr_type, &S.empty_chunk);
        }

        fn chunk_count(self: *Self) u32 {
            return self.chunk_mask + 1;
        }

        fn resetAndShrink(self: *Self, allocator: Allocator) void {
            const orig_chunk = self.chunk_ptr[0..self.chunk_count()];
            self.chunk_ptr = @ptrCast(chunk_ptr_type, &S.empty_chunk);
            self.size = 0;
            self.chunk_mask = 0;

            allocator.free(orig_chunk);
        }

        fn deinit(self: *Self, allocator: Allocator) void {
            self.reset(allocator, true);
        }

        fn resetInplace(self: *Self) void {
            // do not change chunk_mask and chunk_ptr
            // clear all the tag.
            const chunks = self.chunk_ptr;
            const scale = chunks[0].head.get_scale();

            for (0..self.chunk_count()) |idx| {
                const chunk = &chunks[idx];
                chunk.head.clear();
            }
            chunks[0].head.mark_eof(scale);
            self.size = 0;
        }

        fn reset(self: *Self, allocator: Allocator, comptime free: bool) void {
            if (self.is_empty())
                return;

            if (free or self.chunk_count() > 16) {
                self.resetAndShrink(allocator);
                return;
            }

            // keep the size
            self.resetInplace();
        }

        fn new(cap: usize, allocator: Allocator) !Self {
            var map = Self{ .chunk_ptr = @ptrCast(chunk_ptr_type, &S.empty_chunk), .chunk_mask = 0, .size = 0, .flags = 0 };
            if (cap == 0)
                return map;

            try map.reserve(cap, allocator);
            return map;
        }

        const fmap_cap = struct {
            chunk_count: u32,
            scale: u32,
        };

        fn compute_capacity(cap: fmap_cap) u32 {
            return cap.chunk_count * cap.scale;
        }

        fn compute(desired: u32) fmap_cap {
            const minChunks = (desired - 1) / DESIRED_CAP + 1;
            const chunkPow = @intCast(u5, find_last_set(minChunks - 1));

            return .{ .chunk_count = @intCast(u32, 1) << chunkPow, .scale = DESIRED_CAP };
        }

        fn init_chunks(chunks: []chunk_type, cap: fmap_cap) chunk_ptr_type {
            for (0..cap.chunk_count) |idx| {
                const h = &chunks[idx].head;
                h.clear();
            }

            const h = &chunks[0].head;
            h.mark_eof(cap.scale);
            return chunks.ptr;
        }

        const HOSTED_OVERFLOW_INC = @intCast(i8, 0x10);
        const HOSTED_OVERFLOW_DEC = @intCast(i8, -0x10);

        fn alloc_tag(self: *Self, fullness: [*]u8, hp: hash_pair) *item_type {
            var index = hp.hash;
            const delta = hp.probe_delta();
            var hostedOp: i8 = 0;
            var chunk: *chunk_type = undefined;

            while (true) {
                index &= self.chunk_mask;
                chunk = &self.chunk_ptr[index];
                if (fullness[index] < CAPACITY) {
                    break;
                }
                chunk.head.inc_overflow_count();
                hostedOp = HOSTED_OVERFLOW_INC;
                index += delta;
            }

            const item_idx = fullness[index];
            fullness[index] += 1;
            chunk.head.set_tag(item_idx, hp.tag);
            chunk.head.adj_hosted_overflow_count(hostedOp);
            return &chunk.items[item_idx];
        }

        fn erase(itemIter: *item_iter_type) void {
            const chunk = &itemIter.to_chunk()[0];
            chunk.head.clear_tag(itemIter.index);
        }

        fn eraseWithOverflow(self: *Self, itemIter: *item_iter_type, hp: hash_pair) void {
            var hostedOp: i8 = 0;
            var index = hp.hash;
            const delta = hp.probe_delta();
            const chunk = &itemIter.to_chunk()[0];

            while (true) {
                const chunk_ = &self.chunk_ptr[index & self.chunk_mask];
                if (chunk == chunk_) {
                    chunk_.head.clear_tag(itemIter.index);
                    chunk_.head.adj_hosted_overflow_count(hostedOp);
                    break;
                }

                chunk_.head.dec_overflow_count();
                hostedOp = HOSTED_OVERFLOW_DEC;
                index += delta;
            }
        }

        fn remove(self: *Self, itemIter: *item_iter_type) void {
            const chunk_ptr = itemIter.to_chunk();
            const chunk = &chunk_ptr[0];

            if (chunk.head.hosted_overflow_count() != 0) {
                const item = itemIter.item.?[0];
                const hp = hasher.hash(item.getKey());
                self.eraseWithOverflow(itemIter, hp);
            } else {
                erase(itemIter);
            }

            self.size -= 1;
            itemIter.advance_likely_dead();
        }

        fn rehash(self: *Self, orig: fmap_cap, new_cap: fmap_cap, allocator: Allocator) !void {
            const new_chunks = try allocator.alloc(chunk_type, new_cap.chunk_count);
            const orig_chunks = self.chunk_ptr;
            defer {
                // if orig fmap is an empty map, its scale is zero, so its capacity is
                // 0 too.
                if (compute_capacity(orig) > 0)
                    allocator.free(orig_chunks[0..orig.chunk_count]);
            }

            self.chunk_ptr = init_chunks(new_chunks, new_cap);
            self.chunk_mask = new_cap.chunk_count - 1;

            if (self.size == 0) {
                return;
            } else if (orig.chunk_count == 1 and new_cap.chunk_count == 1) {
                var src_idx: usize = 0;
                var dst_idx: usize = 0;
                const src_chunk = &orig_chunks[0];
                const dst_chunk = &new_chunks.ptr[0];

                while (dst_idx < self.size) {
                    // bring scattered items into dense items.
                    if (src_chunk.head.idx_used(src_idx)) {
                        dst_chunk.head.set_tag(dst_idx, src_chunk.head.get_tag(src_idx));
                        dst_chunk.items[dst_idx] = src_chunk.items[src_idx];
                        dst_idx += 1;
                    }
                    src_idx += 1;
                }
            } else {
                var stack_buf: [256]u8 = undefined;
                @memset(&stack_buf, 0);
                var fullness: [*]u8 = &stack_buf;

                if (new_cap.chunk_count > 256) {
                    var mem = try allocator.alloc(u8, new_cap.chunk_count);
                    @memset(mem, 0);
                    fullness = mem.ptr;
                }

                defer {
                    if (fullness != &stack_buf) {
                        allocator.free(fullness[0..new_cap.chunk_count]);
                    }
                }

                //rehash from bottom
                var src_chunks = orig_chunks + orig.chunk_count - 1;
                var remaining = self.size;
                while (remaining > 0) {
                    const src_chunk = &src_chunks[0];
                    var iter = dense_iter.from_chunk_head(src_chunk.head);

                    // prefetch all items here
                    var prefetch_iter = iter;
                    while (prefetch_iter.next()) |idx| {
                        @prefetch(&src_chunk.items[idx], .{});
                    }

                    // begin rehash
                    while (iter.next()) |src_idx| {
                        remaining -= 1;
                        const src_item = &src_chunk.items[src_idx];
                        const hp = hasher.hash(src_item.getKey());
                        const item_ptr = self.alloc_tag(fullness, hp);
                        item_ptr.* = src_item.*;
                    }
                    src_chunks -= 1;
                }
            }
        }

        fn reserve(self: *Self, cap: usize, allocator: Allocator) !void {
            const desired = @intCast(u32, @max(self.size, cap));
            if (desired == 0) {
                self.reset(allocator, true);
                return;
            }

            const h = self.chunk_ptr[0].head;
            const orig = fmap_cap{ .chunk_count = self.chunk_mask + 1, .scale = h.get_scale() };

            const orig_size = compute_capacity(orig);

            if (desired <= orig_size and
                desired >= orig_size - orig_size / 8)
            {
                return;
            }

            const new_cap = compute(desired);
            const new_size = compute_capacity(new_cap);

            if (new_size != orig_size) {
                try self.rehash(orig, new_cap, allocator);
            }
        }

        fn findImpl(self: *Self, hp: hash_pair, item: *const item_type, comptime prefetch: bool) item_iter_type {
            var index = hp.hash;
            var tries: usize = 0;
            const step = hp.probe_delta();

            while (tries <= self.chunk_mask) : (tries += 1) {
                const chunk = &self.chunk_ptr[index & self.chunk_mask];
                if (prefetch) {
                    @prefetch(&chunk.items, .{});
                }

                var iter = match_iter.new(chunk.head, @intCast(u8, hp.tag));
                while (iter.next()) |idx| {
                    if (item.keyEql(&chunk.items[idx])) {
                        return item_iter_type.new(chunk, idx);
                    }
                }

                if (chunk.head.overflow_count() == 0) {
                    break;
                }
                index += step;
            }

            return item_iter_type{ .item = null, .index = 0 };
        }

        fn find(self: *Self, item: *const item_type) item_iter_type {
            const hp = hasher.hash(item.getKey());
            return self.findImpl(hp, item, true);
        }

        fn get_scale(self: *Self) u32 {
            return self.chunk_ptr[0].head.get_scale();
        }

        fn reserveForInsertImpl(self: *Self, cap_minus_one: u32, orig_cap: fmap_cap, cap: u32, allocator: Allocator) !void {
            const needed_cap = cap_minus_one + 1;

            //we want to grow by between 2^0.5 and 2^1.5 ending at a "good"
            // size, so we grow by 2^0.5 and then round up
            // 1.01101_2 = 1.40625
            const min_growth = cap + (cap >> 2) + (cap >> 3) + (cap >> 5);

            const desired = @max(needed_cap, min_growth);
            const new_cap = compute(desired);
            try self.rehash(orig_cap, new_cap, allocator);
        }

        fn reserve_for_insert(self: *Self, incoming: u32, allocator: Allocator) !void {
            const needed = self.size + incoming;
            const scale = self.get_scale();
            const existing_cap = compute_capacity(.{ .chunk_count = self.chunk_count(), .scale = scale });

            if (needed - 1 >= existing_cap) {
                try self.reserveForInsertImpl(needed - 1, .{ .chunk_count = self.chunk_count(), .scale = scale }, existing_cap, allocator);
            }
        }

        fn insert(self: *Self, item: *const item_type, allocator: Allocator) !void {
            const hp = hasher.hash(item.getKey());
            if (self.size > 0) {
                // check if the key exists?
                const iter = self.findImpl(hp, item, true);
                if (!iter.at_end()) {
                    return error.ItemExist;
                }
            }

            try self.reserve_for_insert(1, allocator);

            var index = hp.hash;
            var chunk = &self.chunk_ptr[index & self.chunk_mask];
            const idx = chunk.head.first_empty_idx() orelse blk: {
                const delta = hp.probe_delta();
                while (true) {
                    chunk.head.inc_overflow_count();
                    index += delta;
                    chunk = &self.chunk_ptr[index & self.chunk_mask];

                    if (chunk.head.first_empty_idx()) |this_idx| {
                        chunk.head.adj_hosted_overflow_count(HOSTED_OVERFLOW_INC);
                        break :blk this_idx;
                    }
                }
            };

            chunk.head.set_tag(idx, hp.tag);
            chunk.items[idx] = item.*;

            self.size += 1;
        }
    };
}

fn getHasher(comptime K: type) type {
    return struct {
        fn hash(key: K) hash_pair {
            const hv = std.hash.Wyhash.hash(0, std.mem.asBytes(&key));
            return hash_pair.from_hash(@truncate(u32, hv));
        }
    };
}

fn getItemTypeKeyValue(comptime Key: type, comptime Value: type) type {
    return extern struct {
        key: Key,
        value: Value,

        const keyType = Key;
        const valueType = Value;

        fn getKey(self: @This()) Key {
            return self.key;
        }

        fn getValue(self: @This()) *Value {
            return &self.value;
        }
    };
}

fn getItemTypeKeyOnly(comptime Key: type) type {
    return extern struct {
        key: Key,

        const keyType = Key;

        fn getKey(self: @This()) Key {
            return self.key;
        }

        fn keyEql(self: @This(), other: *@This()) bool {
            return std.meta.eql(self.getKey(), other.getKey());
        }
    };
}

test {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const itemType = getItemTypeKeyOnly(u64);
    const hasher = getHasher(itemType.keyType);

    var map = try fmap(itemType, hasher).new(0, allocator);
    try testing.expect(map.chunk_mask == 0);
    map.deinit(allocator);

    const status = gpa.deinit();
    try testing.expect(status != .leak);
}

test {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const itemType = getItemTypeKeyOnly(u64);
    const hasher = getHasher(itemType.keyType);

    var map = try fmap(itemType, hasher).new(100, allocator);
    try testing.expect(map.chunk_mask == 15);
    map.deinit(allocator);

    const status = gpa.deinit();
    try testing.expect(status != .leak);
}

test {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const itemType = getItemTypeKeyOnly(u64);
    const hasher = getHasher(itemType.keyType);

    var map = try fmap(itemType, hasher).new(0, allocator);

    const item = itemType{ .key = 1 };
    try map.insert(&item, allocator);
    var iter = map.find(&item);
    try testing.expect(!iter.at_end());
    try testing.expect(iter.index == 0);

    map.remove(&iter);
    try testing.expect(iter.at_end());

    map.deinit(allocator);

    const status = gpa.deinit();
    try testing.expect(status != .leak);
}

test "fmap a large insert and remove" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const itemType = getItemTypeKeyOnly(u64);
    const hasher = getHasher(itemType.keyType);

    var map = try fmap(itemType, hasher).new(0, allocator);

    for (0..1024) |idx| {
        const item = itemType{ .key = idx };
        try map.insert(&item, allocator);
    }
    try testing.expect(map.size == 1024);

    for (0..map.chunk_count()) |idx| {
        // make sure there is no overflow in overflow_count.
        try testing.expect(map.chunk_ptr[idx].head.overflow_count() < 128);
        if (idx != 0) {
            try testing.expect(map.chunk_ptr[idx].head.get_scale() == 0);
        }
    }

    for (0..1024) |idx| {
        const item = itemType{ .key = idx };
        var iter = map.find(&item);
        try testing.expect(iter.item.?[0].key == idx);
        try testing.expect(!iter.at_end());
        map.remove(&iter);

        if (idx == 1023) {
            try testing.expect(iter.item == null);
        }
    }
    try testing.expect(map.size == 0);

    for (0..map.chunk_count()) |idx| {
        // make sure there is no overflow in overflow_count.
        try testing.expect(map.chunk_ptr[idx].head.overflow_count() < 128);
        if (idx != 0) {
            try testing.expect(map.chunk_ptr[idx].head.get_scale() == 0);
        }
    }

    map.deinit(allocator);

    const status = gpa.deinit();
    try testing.expect(status != .leak);
}
