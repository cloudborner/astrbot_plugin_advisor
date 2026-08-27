import asyncio
import tempfile
import unittest
from pathlib import Path

from PIL import Image as PILImage

from advisor.analysis_draft import DraftImage
from advisor.image_evidence import (
    _assert_public_remote_url,
    cleanup_prepared_images,
    prepare_images,
    validate_remote_images,
)


def image(index: int, reference: str) -> DraftImage:
    return DraftImage(
        evidence_id=f"图片{index:03d}",
        reference=reference,
        message_evidence_id=f"消息{index:04d}",
        timestamp=index,
    )


class ImageEvidenceTests(unittest.TestCase):
    def test_remote_tokens_do_not_defeat_deduplication(self):
        result = prepare_images(
            [
                image(1, "https://example.com/a.jpg?token=old&file=1"),
                image(2, "https://example.com/a.jpg?token=new&file=1"),
            ],
            maximum=8,
        )
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.duplicate_count, 1)

    def test_local_content_hash_deduplicates_different_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.png"
            second = Path(directory) / "b.png"
            content = b"\x89PNG\r\n\x1a\n" + b"same-content"
            first.write_bytes(content)
            second.write_bytes(content)
            result = prepare_images(
                [image(1, str(first)), image(2, str(second))], maximum=8
            )
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.duplicate_count, 1)

    def test_invalid_input_is_skipped_without_losing_valid_images(self):
        result = prepare_images(
            [image(1, "not-a-file"), image(2, "https://example.com/valid.jpg")],
            maximum=8,
        )
        self.assertEqual([item.evidence_id for item in result.images], ["图片002"])
        self.assertEqual(result.invalid_count, 1)

    def test_sampling_spans_the_whole_time_range(self):
        result = prepare_images(
            [image(index, f"https://example.com/{index}.jpg") for index in range(1, 11)],
            maximum=3,
        )
        self.assertEqual(
            [item.evidence_id for item in result.images],
            ["图片001", "图片005", "图片010"],
        )

    def test_sampling_prefers_text_adjacent_image_near_each_timeline_anchor(self):
        rows = [image(index, f"https://example.com/{index}.jpg") for index in range(1, 10)]
        rows[1] = DraftImage(
            evidence_id="图片002",
            reference="https://example.com/2.jpg",
            message_evidence_id="消息0002",
            timestamp=2,
            context_weight=3,
        )
        result = prepare_images(rows, maximum=3)
        self.assertEqual(
            [item.evidence_id for item in result.images],
            ["图片002", "图片005", "图片009"],
        )

    def test_remote_content_hash_deduplicates_different_urls(self):
        preliminary = prepare_images(
            [
                image(1, "https://example.com/a.jpg"),
                image(2, "https://cdn.example.com/copy.jpg"),
            ],
            maximum=8,
        )

        def probe(_reference: str, _timeout: float, _maximum: int):
            return "content:" + "a" * 64, 128

        result = asyncio.run(
            validate_remote_images(preliminary, maximum=8, probe=probe)
        )
        self.assertEqual([item.evidence_id for item in result.images], ["图片001"])
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(result.checked_remote_count, 2)
        self.assertEqual(result.downloaded_bytes, 256)

    def test_remote_partial_failure_keeps_other_images(self):
        preliminary = prepare_images(
            [
                image(1, "https://example.com/bad.jpg"),
                image(2, "https://example.com/good.jpg"),
            ],
            maximum=8,
        )

        def probe(reference: str, _timeout: float, _maximum: int):
            if "bad" in reference:
                return None
            return "content:" + "b" * 64, 256

        result = asyncio.run(
            validate_remote_images(preliminary, maximum=8, probe=probe)
        )
        self.assertEqual([item.evidence_id for item in result.images], ["图片002"])
        self.assertEqual(result.invalid_count, 1)

    def test_oversized_local_image_is_resized_and_temporary_file_is_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wide.png"
            with PILImage.new("RGB", (3000, 40), "white") as created:
                created.save(source, "PNG")
            preliminary = prepare_images([image(1, str(source))], maximum=8)
            result = asyncio.run(validate_remote_images(preliminary, maximum=8))
            self.assertEqual(len(result.images), 1)
            resized = Path(result.images[0].reference)
            self.assertNotEqual(resized, source)
            self.assertTrue(resized.is_file())
            with PILImage.open(resized) as checked:
                self.assertLessEqual(max(checked.size), 2048)
            cleanup_prepared_images(result)
            self.assertFalse(resized.exists())
            cleanup_prepared_images(result)

    def test_private_and_credentialed_remote_urls_are_rejected(self):
        for reference in (
            "http://127.0.0.1/a.png",
            "http://[::1]/a.png",
            "https://user:password@example.com/a.png",
            "https://example.com:8080/a.png",
        ):
            with self.assertRaises(ValueError):
                _assert_public_remote_url(reference)


if __name__ == "__main__":
    unittest.main()
