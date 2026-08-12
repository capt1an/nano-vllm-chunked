from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.last_access_time = 0
        
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.last_access_time = 0


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set() # ref_count=0 hash=-1
        self.cached_block_ids: set[int] = set() # ref_count=0 hash!=-1
        self.time = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        if len(self.free_block_ids)==0:
            self._evict_block()
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        #     del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        # block = self.blocks[block_id]
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        block = self.blocks[block_id]
        # deallocate 的请求block难道会是一个没有hash的block吗？应该只需要添加到cached_block_ids中就行了。
        # 并不是这样，因为涉及到prefix cache的最后一个未满的block，
        if block.hash != -1: # hash ！= -1 表示是一个占满的block，可以被用来prefix
            self.cached_block_ids.add(block_id)
        else: # hash = -1 表示是一个没有占满的block，直接释放掉就行了
            self.free_block_ids.append(block_id)

    # lru eviction
    def _evict_block(self):
        if not self.cached_block_ids:
            return None
        victim = min(self.cached_block_ids, key=lambda x: self.blocks[x].last_access_time)
        block = self.blocks[victim]
        del self.hash_to_block_id[block.hash]
        self.cached_block_ids.remove(victim)
        # block.reset() 
        self.free_block_ids.append(victim)
        # print("victim idx:", victim, "hash:", block.hash, "last_access_time:", block.last_access_time)
        

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # print("seq.num_blocks", seq.num_blocks)
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # if block_id in self.used_block_ids:
            #     num_new_blocks -= 1
        
        num_new_blocks = seq.num_blocks - num_cached_blocks
        if len(self.free_block_ids) + len(self.cached_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        # print("开始allocate", num_cached_blocks)
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.cached_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
            self.time += 1
            block.last_access_time = self.time
            # print("触发prefix cache !")
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) + len(self.cached_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
            self.time+=1
            block.last_access_time=self.time


    def print_status(self):
        print("=" * 80)
        print("Block Manager Status")
        print(f"Total blocks: {len(self.blocks)}")
        print(f"Used blocks: {len(self.used_block_ids)}")
        print(f"Free blocks: {len(self.free_block_ids)}")
        print(f"Cached blocks: {sum(1 for b in self.blocks if b.hash != -1 and b.ref_count == 0)}")
        print("-" * 80)

        print("free_block_ids:")
        print(list(self.free_block_ids))

        print("\nused_block_ids:")
        print(sorted(list(self.used_block_ids)))

        print("-" * 80)
        print("Block details:")

        for block in self.blocks:
            if (
                block.block_id in self.used_block_ids
                or block.hash != -1
                or block.ref_count > 0
            ):
                if block.block_id in self.used_block_ids:
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
                    f"last_access={block.last_access_time} | "
                    f"tokens={block.token_ids[:8]}"
                    f"{'...' if len(block.token_ids) > 8 else ''}"
                )

        print("-" * 80)

        print("hash_to_block_id:")
        for h, block_id in self.hash_to_block_id.items():
            print(f"{h} -> block {block_id}")

        print("=" * 80)