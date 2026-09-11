from collections import OrderedDict

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class EvictableBlockQueue:
    """An O(1) queue for every block whose reference count is zero.

    Cached blocks and blocks without reusable contents intentionally share the
    same queue. Cached blocks are kept in LRU order at the back; uncached
    blocks are returned to the front so they can be reused without evicting a
    useful prefix.
    """

    def __init__(self, block_ids):
        self._block_ids = OrderedDict.fromkeys(block_ids)

    def __len__(self):
        return len(self._block_ids)

    def __iter__(self):
        return iter(self._block_ids)

    def popleft(self) -> int:
        if not self._block_ids:
            raise RuntimeError("No evictable KV cache block is available")
        block_id, _ = self._block_ids.popitem(last=False)
        return block_id

    def append(self, block_id: int):
        assert block_id not in self._block_ids
        self._block_ids[block_id] = None

    def appendleft(self, block_id: int):
        self.append(block_id)
        self._block_ids.move_to_end(block_id, last=False)

    def remove(self, block_id: int):
        del self._block_ids[block_id]

    def __contains__(self, block_id: int):
        return block_id in self._block_ids


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # Duplicate physical blocks can contain the same prefix when requests
        # are computed together, so one hash may map to multiple block IDs.
        self.hash_to_block_ids: dict[int, set[int]] = {}
        self.free_block_ids = EvictableBlockQueue(range(num_blocks))
        self.reserved_block_ids: dict[int, list[int]] = {}

    def allocate_exclusive(self, seq: Sequence, total_blocks: int) -> bool:
        """Reserve a PD request's maximum footprint without publishing cache hits.

        Future decode blocks are held separately so block_table only describes
        the current sequence. Admission either reserves everything or changes nothing.
        """
        if seq.block_table or seq.seq_id in self.reserved_block_ids:
            raise ValueError("Sequence already has KV blocks")
        if total_blocks < seq.num_blocks:
            raise ValueError("Reservation cannot be shorter than the prompt")
        if total_blocks > len(self.free_block_ids):
            return False
        blocks = [self._allocate_block() for _ in range(total_blocks)]
        seq.block_table.extend(blocks[:seq.num_blocks])
        self.reserved_block_ids[seq.seq_id] = blocks[seq.num_blocks:]
        return True

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _cache_block(self, block: Block):
        self.hash_to_block_ids.setdefault(block.hash, set()).add(block.block_id)

    def _uncache_block(self, block: Block):
        if block.hash == -1:
            return
        block_ids = self.hash_to_block_ids.get(block.hash)
        if block_ids is not None:
            block_ids.discard(block.block_id)
            if not block_ids:
                del self.hash_to_block_ids[block.hash]
        block.hash = -1
        block.token_ids = []

    def _find_cached_block(self, block_hash: int, token_ids: list[int]):
        idle_match = None
        for block_id in self.hash_to_block_ids.get(block_hash, ()):
            block = self.blocks[block_id]
            if block.token_ids == token_ids:
                # Sharing an already used block consumes no evictable block,
                # so prefer it when duplicate physical copies exist.
                if block.ref_count > 0:
                    return block
                idle_match = block
        return idle_match

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self._uncache_block(block)
        block.reset()
        return block_id

    def _touch_block(self, block: Block):
        if block.ref_count == 0:
            self.free_block_ids.remove(block.block_id)
        block.ref_count += 1

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_used_hits = 0
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block = self._find_cached_block(h, token_ids)
            if block is None:
                break
            num_cached_blocks += 1
            if block.ref_count > 0:
                num_used_hits += 1

        # A cached block with ref_count == 0 is still in the free queue. It
        # must be reserved by this request and cannot also fund a new block.
        num_required_blocks = seq.num_blocks - num_used_hits
        if len(self.free_block_ids) < num_required_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block = self._find_cached_block(h, token_ids)
            assert block is not None
            self._touch_block(block)
            seq.block_table.append(block.block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        # Releasing in reverse order makes request-specific suffix blocks
        # eviction candidates before the more reusable prefix blocks.
        blocks = seq.block_table + self.reserved_block_ids.pop(seq.seq_id, [])
        for block_id in reversed(blocks):
            block = self.blocks[block_id]
            assert block.ref_count > 0
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.hash == -1:
                    self.free_block_ids.appendleft(block_id)
                else:
                    self.free_block_ids.append(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return (len(seq.block_table) >= seq.num_blocks
                or bool(self.reserved_block_ids.get(seq.seq_id))
                or bool(len(self.free_block_ids)))

    def may_append(self, seq: Sequence):
        if len(seq.block_table) < seq.num_blocks:
            reserved = self.reserved_block_ids.get(seq.seq_id)
            seq.block_table.append(reserved.pop() if reserved else self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self._cache_block(block)

    def print_status(self):
        print("=" * 80)
        print("Block Manager Status")
        print(f"Total blocks: {len(self.blocks)}")
        print(f"Used blocks: {sum(block.ref_count > 0 for block in self.blocks)}")
        print(f"Evictable blocks: {len(self.free_block_ids)}")
        print(f"Cached blocks: {sum(block.hash != -1 for block in self.blocks)}")
        print("-" * 80)
        print("Eviction order (first to last):")
        print(list(self.free_block_ids))
        print("-" * 80)
        print("Block details:")
        for block in self.blocks:
            if block.ref_count > 0:
                state = "USED"
            elif block.hash != -1:
                state = "CACHED"
            else:
                state = "FREE"
            print(
                f"Block {block.block_id:4d} | "
                f"{state:6s} | "
                f"ref={block.ref_count} | "
                f"hash={block.hash} | "
                f"tokens={block.token_ids[:8]}"
                f"{'...' if len(block.token_ids) > 8 else ''}"
            )
        print("-" * 80)
        print("hash_to_block_ids:")
        for h, block_ids in self.hash_to_block_ids.items():
            print(f"{h} -> blocks {sorted(block_ids)}")
        print("=" * 80)
