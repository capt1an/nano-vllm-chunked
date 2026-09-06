import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class BlockManagerTest(unittest.TestCase):

    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 2

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    @staticmethod
    def allocate_and_hash(manager: BlockManager, token_ids: list[int]):
        seq = Sequence(token_ids)
        num_cached_blocks = manager.can_allocate(seq)
        if num_cached_blocks < 0:
            raise AssertionError("test sequence does not fit")
        manager.allocate(seq, num_cached_blocks)
        seq.num_scheduled_tokens = len(token_ids) - seq.num_cached_tokens
        manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        return seq

    def assert_invariants(self, manager: BlockManager):
        evictable = set(manager.free_block_ids)
        self.assertEqual(len(evictable), len(manager.free_block_ids))
        self.assertEqual(
            evictable,
            {block.block_id for block in manager.blocks if block.ref_count == 0},
        )
        for block_hash, block_ids in manager.hash_to_block_ids.items():
            for block_id in block_ids:
                self.assertEqual(manager.blocks[block_id].hash, block_hash)

    def test_idle_cache_hits_are_reserved_during_capacity_check(self):
        manager = BlockManager(num_blocks=2, block_size=2)
        cached = self.allocate_and_hash(manager, [1, 2, 3, 4])
        manager.deallocate(cached)

        request = Sequence([1, 2, 3, 4, 5])
        self.assertEqual(manager.can_allocate(request), -1)
        self.assert_invariants(manager)

    def test_release_orders_suffix_before_prefix_for_eviction(self):
        manager = BlockManager(num_blocks=3, block_size=2)
        seq = self.allocate_and_hash(manager, [1, 2, 3, 4, 5, 6])
        block_table = list(seq.block_table)

        manager.deallocate(seq)

        self.assertEqual(list(manager.free_block_ids), list(reversed(block_table)))
        victim = manager._allocate_block()
        self.assertEqual(victim, block_table[-1])
        self.assert_invariants(manager)

    def test_cache_hit_moves_block_to_back_after_release(self):
        manager = BlockManager(num_blocks=3, block_size=2)
        original = self.allocate_and_hash(manager, [1, 2, 3, 4])
        prefix_block = original.block_table[0]
        suffix_block = original.block_table[1]
        manager.deallocate(original)

        hit = Sequence([1, 2, 9])
        self.assertEqual(manager.can_allocate(hit), 1)
        manager.allocate(hit, 1)
        self.assertEqual(hit.block_table[0], prefix_block)
        manager.deallocate(hit)

        eviction_order = list(manager.free_block_ids)
        self.assertLess(
            eviction_order.index(suffix_block),
            eviction_order.index(prefix_block),
        )
        self.assertEqual(eviction_order[-1], prefix_block)
        self.assert_invariants(manager)

    def test_duplicate_hash_blocks_are_indexed_and_evicted_independently(self):
        manager = BlockManager(num_blocks=2, block_size=2)
        first = Sequence([1, 2])
        second = Sequence([1, 2])

        # Allocate both before hashing to model identical requests in one batch.
        manager.allocate(first, manager.can_allocate(first))
        manager.allocate(second, manager.can_allocate(second))
        for seq in (first, second):
            seq.num_scheduled_tokens = 2
            manager.hash_blocks(seq)

        block_hash = manager.blocks[first.block_table[0]].hash
        self.assertEqual(manager.hash_to_block_ids[block_hash], {0, 1})
        manager.deallocate(first)
        manager.deallocate(second)

        first_victim = manager._allocate_block()
        self.assertIn(block_hash, manager.hash_to_block_ids)
        self.assertEqual(len(manager.hash_to_block_ids[block_hash]), 1)
        second_victim = manager._allocate_block()
        self.assertNotEqual(first_victim, second_victim)
        self.assertNotIn(block_hash, manager.hash_to_block_ids)
        self.assert_invariants(manager)

    def test_used_duplicate_is_preferred_without_consuming_capacity(self):
        manager = BlockManager(num_blocks=2, block_size=2)
        first = Sequence([1, 2])
        second = Sequence([1, 2])
        manager.allocate(first, 0)
        manager.allocate(second, 0)
        for seq in (first, second):
            seq.num_scheduled_tokens = 2
            manager.hash_blocks(seq)
        manager.deallocate(second)

        request = Sequence([1, 2, 3])
        self.assertEqual(manager.can_allocate(request), 1)
        manager.allocate(request, 1)
        self.assertEqual(request.block_table[0], first.block_table[0])
        self.assert_invariants(manager)


if __name__ == "__main__":
    unittest.main()
