import unittest

from src.classifier.sliding_window import chunk_token_ids


class ChunkTokenIdsTests(unittest.TestCase):
    def test_short_content_returns_single_unchanged_window(self) -> None:
        content = [1, 2, 3]
        windows = chunk_token_ids(content, max_length=8, stride=4)
        self.assertEqual(windows, [[1, 2, 3]])

    def test_content_exactly_at_window_size_returns_single_window(self) -> None:
        # max_length=6 -> window_size=4
        content = [1, 2, 3, 4]
        windows = chunk_token_ids(content, max_length=6, stride=2)
        self.assertEqual(windows, [[1, 2, 3, 4]])

    def test_empty_content_returns_single_empty_window(self) -> None:
        self.assertEqual(chunk_token_ids([], max_length=8, stride=4), [[]])

    def test_long_content_overlaps_and_covers_everything(self) -> None:
        # max_length=6 -> window_size=4, stride=2 -> 2-token overlap
        content = list(range(10))
        windows = chunk_token_ids(content, max_length=6, stride=2)
        self.assertEqual(
            windows,
            [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]],
        )
        covered = {token for window in windows for token in window}
        self.assertEqual(covered, set(content))

    def test_last_window_may_be_shorter_than_window_size(self) -> None:
        # window_size=4, stride=3 -> windows start at 0, 3, 6 -> last is [6..8], len 3
        content = list(range(9))
        windows = chunk_token_ids(content, max_length=6, stride=3)
        self.assertEqual(windows[-1], [6, 7, 8])

    def test_rejects_stride_not_smaller_than_window_size(self) -> None:
        content = list(range(20))
        with self.assertRaises(ValueError):
            chunk_token_ids(content, max_length=6, stride=4)  # stride == window_size
        with self.assertRaises(ValueError):
            chunk_token_ids(content, max_length=6, stride=0)

    def test_rejects_max_length_too_small_for_special_tokens(self) -> None:
        with self.assertRaises(ValueError):
            chunk_token_ids([1, 2, 3], max_length=2, stride=1)


if __name__ == "__main__":
    unittest.main()
