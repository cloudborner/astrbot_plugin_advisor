import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_extracted_sources as scanner  # noqa: E402


def plugin_for(path: Path) -> dict:
    return {"plugin_id": "test/plugin", "repo": "https://github.com/test/plugin", "version": "1.0", "commit_sha": "a" * 40, "source_dir": path.name}


class ScannerUnitTests(unittest.TestCase):
    def test_exclusions_and_lockfile_do_not_trigger_transformers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("from astrbot.core import llm\n\ndef run():\n    return llm\n", encoding="utf-8")
            (root / "package-lock.json").write_text('{"@shikijs/transformers": {}}', encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs" / "fake.py").write_text("import transformers\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertNotIn("local_model", profile["features"])
            self.assertNotIn("transformers", profile["dependencies"])

    def test_spectre_remote_llm_and_global_cache_without_local_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("from .utils import ImageCaptionUtils\n\nasync def on_event(ctx):\n    return ctx.get_using_provider()\n", encoding="utf-8")
            (root / "utils").mkdir()
            (root / "utils" / "__init__.py").write_text("from .image_caption import ImageCaptionUtils\n", encoding="utf-8")
            (root / "utils" / "image_caption.py").write_text("class ImageCaptionUtils:\n    caption_cache = {}\n    async def caption(self, ctx):\n        self.caption_cache[ctx] = 'caption'\n        return await ctx.get_provider_by_id('x')\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertNotIn("local_model", profile["features"])
            self.assertTrue(profile["cache"]["unbounded_growth_possible"])
            self.assertEqual(profile["scores"]["peak_memory"], 3)
            self.assertTrue(any(e["kind"] == "remote_api" for e in profile["evidence"]))

    def test_downloader_detects_ffmpeg_size_limit_and_bounded_semaphore(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text(
                "import asyncio, subprocess\n"
                "sem = asyncio.Semaphore(3)\n"
                "max_size = 30\n"
                "async def run(url):\n"
                "    async with sem:\n"
                "        subprocess.run(['yt-dlp', url])\n"
                "        subprocess.run(['ffmpeg', '-i', 'in', 'out'])\n"
                "        with open('out.mp4', 'rb') as f:\n"
                "            return f.read(1024 * 1024)\n"
                "        return await upload_file('out.mp4')\n",
                encoding="utf-8",
            )
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertTrue(profile["concurrency"]["bounded"])
            self.assertIn("ffmpeg", profile["external_processes"])
            self.assertIn("yt-dlp", profile["runtime_downloads"])
            self.assertIn("download_limit", profile["features"])
            self.assertIn("large_upload", profile["features"])
            self.assertTrue(profile["background_tasks"]["bounded"])
            self.assertEqual(profile["scores"]["peak_memory"], 0)
            self.assertGreaterEqual(profile["scores"]["peak_cpu"], 3)

    def test_jm_full_read_raises_peak_memory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text(
                "import asyncio\n"
                "sem = asyncio.Semaphore(2)\n"
                "async def download():\n"
                "    with open('book.pdf', 'rb') as file_handle:\n"
                "        data = file_handle.read()\n"
                "    return data\n",
                encoding="utf-8",
            )
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertTrue(profile["concurrency"]["bounded"])
            self.assertGreaterEqual(profile["scores"]["peak_memory"], 2)
            self.assertEqual(profile["runtime_downloads"], [])

    def test_invalid_ast_isolated_and_large_binary_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
            (root / "broken.py").write_text("def broken(:\n", encoding="utf-8")
            (root / "huge.py").write_bytes(b"x" * (scanner.MAX_TEXT_BYTES + 1))
            (root / "blob.py").write_bytes(b"\x00\x01\x02")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertIn("broken.py: AST SyntaxError", " ".join(profile["unknowns"]))
            self.assertIn("huge.py: file exceeds safe AST size limit", " ".join(profile["unknowns"]))
            self.assertIn("blob.py: binary file skipped", " ".join(profile["unknowns"]))
            self.assertIn("main.py", profile["reachable_python_files"])

    def test_actual_regression_directories_when_available(self):
        source = ROOT / "source_extracted"
        samples = {
            "23q3__astrbot_plugin_SpectreCore__d6076ef9b972": ("remote_llm", "cache"),
            "204343414__astrbot_plugin_yt-dlp__2ba4ce593ea8": ("yt-dlp", "ffmpeg"),
            "lxfight-s-Astrbot-Plugins__astrbot_plugin_livingmemory__c2e733049392": ("storage", "remote_llm"),
            "CLRedfield__astrbot-plugin-jmdownloader__9172123656f0": ("jmcomic", "pdf"),
        }
        if not source.exists():
            self.skipTest("extracted source set not present")
        for dirname, needles in samples.items():
            path = source / dirname
            self.assertTrue(path.exists(), dirname)
            profile = scanner.SourceAnalyzer(path, plugin_for(path)).profile()
            haystack = " ".join(profile["features"] + profile["runtime_downloads"] + profile["external_processes"])
            for needle in needles:
                if needle == "cache":
                    self.assertTrue(profile["cache"]["present"], dirname)
                else:
                    self.assertIn(needle, haystack, dirname)

    def test_output_contract_and_queue_bound(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source_extracted"
            source.mkdir()
            p = source / "owner__repo__aaaaaaaaaaaa"
            p.mkdir()
            (p / "main.py").write_text("print('not executed')\n", encoding="utf-8")
            (root / "data").mkdir()
            (root / ".cache").mkdir()
            market = {"$meta": {}, "plugins": {"owner/repo": {"repo": "https://github.com/owner/repo", "version": "1", "download_count": 1, "stars": 1}}}
            old = {"$meta": {}, "profiles": {"owner/repo": {"scores": {"peak_memory": 0, "peak_cpu": 0}}}}
            cache = {"owner/repo": {"repo": "https://github.com/owner/repo", "observation": {"commit_sha": "a" * 40}}}
            (root / "data" / "market_snapshot.json").write_text(json.dumps(market), encoding="utf-8")
            (root / "data" / "resource_profiles.json").write_text(json.dumps(old), encoding="utf-8")
            (root / ".cache" / "github_observations.json").write_text(json.dumps(cache), encoding="utf-8")
            args = type("Args", (), {"root": str(root), "source": str(source), "profiles": str(root / "profiles.json"), "queue": str(root / "queue.json")})()
            profiles, queue = scanner.run(args)
            self.assertEqual(profiles["$meta"]["mapping"]["mapped_dir_count"], 1)
            self.assertLessEqual(len(queue["items"]), 30)
            p = next(iter(profiles["profiles"].values()))
            self.assertEqual(p["overall_level"], "L" + str(max(p["scores"]["peak_memory"], p["scores"]["peak_cpu"])))
            self.assertTrue(all(0 <= v <= 4 for v in p["scores"].values()))
            self.assertTrue(all(1 <= e["line"] for e in p["evidence"]))
            profiles2, queue2 = scanner.run(args)
            def normalized(value):
                data = json.loads(json.dumps(value))
                if "$meta" in data:
                    data["$meta"].pop("generated_at", None)
                return data
            self.assertEqual(normalized(profiles), normalized(profiles2))
            self.assertEqual(normalized(queue), normalized(queue2))

    def test_model_file_hint_and_quiet_zero_are_conservative(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("MODEL = 'weights/model.gguf'\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertNotIn("local_model", profile["features"])
            self.assertIn("model_file_reference", profile["features"])
            self.assertEqual(profile["scores"]["peak_memory"], 0)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("import transformers\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertNotIn("local_model", profile["features"])
            self.assertIn("model_dependency", profile["features"])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("def noop():\n    return None\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertEqual(profile["scores"]["peak_memory"], 0)
            self.assertEqual(profile["confidence"], 0.65)

    def test_generic_model_and_tool_strings_are_not_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text(
                "MODEL = 'weights/model.onnx'\n"
                "USER_AGENT = 'Mozilla Chrome/120'\n"
                "ROUTE = '/download/media'\n"
                "def run(service, logger):\n"
                "    logger.info('install ffmpeg then call generate')\n"
                "    return service.load_model('remote-id')\n",
                encoding="utf-8",
            )
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertNotIn("local_model", profile["features"])
            self.assertNotIn("browser", profile["features"])
            self.assertEqual(profile["external_processes"], [])
            self.assertEqual(profile["runtime_downloads"], [])

    def test_actual_model_loaders_distinguish_small_and_heavy_risk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("import onnxruntime as ort\nsession = ort.InferenceSession('small.onnx')\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertIn("local_model", profile["features"])
            self.assertEqual(profile["overall_level"], "L3")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("from transformers import AutoModelForCausalLM\nmodel = AutoModelForCausalLM.from_pretrained('large')\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertIn("local_model", profile["features"])
            self.assertEqual(profile["overall_level"], "L4")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("from sentence_transformers import SentenceTransformer\nmodel = SentenceTransformer('small')\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertIn("local_model", profile["features"])
            self.assertEqual(profile["overall_level"], "L3")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("import local_tts as ts\nmodel = ts.initialize_model('weights')\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertIn("local_model", profile["features"])
            self.assertEqual(profile["overall_level"], "L4")

    def test_browser_requires_launch_not_dependency_or_user_agent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("from playwright.async_api import async_playwright\nUA = 'Chrome/120'\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertNotIn("browser", profile["features"])
            self.assertIn("browser_dependency", profile["features"])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("from playwright.async_api import async_playwright\nasync def run(p):\n    return await p.chromium.launch()\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertIn("browser", profile["features"])
            self.assertEqual(profile["overall_level"], "L3")

    def test_cache_needs_runtime_growth_and_respects_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text(
                "MOTIVATIONAL_MESSAGES = ['a', 'b']\n"
                "schema_cache = {}\n"
                "schema_cache['field'] = str\n",
                encoding="utf-8",
            )
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertFalse(profile["cache"]["unbounded_growth_possible"])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text(
                "cache_ttl = 60\ncaption_cache = {}\n"
                "def put(key, value):\n    caption_cache[key] = value\n",
                encoding="utf-8",
            )
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertTrue(profile["cache"]["bounded"])
            self.assertLess(profile["scores"]["peak_memory"], 3)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text(
                "class Helper:\n"
                "    def __init__(self):\n        self.cache = {}\n"
                "    def put(self, key, value):\n        self.cache[key] = value\n",
                encoding="utf-8",
            )
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertFalse(profile["cache"]["unbounded_growth_possible"])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text(
                "class Star: pass\n"
                "class Main(Star):\n"
                "    def __init__(self):\n        self.messages = []\n"
                "    def remember(self, value):\n        self.messages.append(value)\n",
                encoding="utf-8",
            )
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertTrue(profile["cache"]["unbounded_growth_possible"])

    def test_background_sleep_is_bounded_but_tight_task_producer_is_not(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("import asyncio\nasync def loop():\n    while True:\n        await asyncio.sleep(1)\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertTrue(profile["background_tasks"]["bounded"])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("import asyncio\nasync def loop():\n    while True:\n        asyncio.create_task(work())\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertFalse(profile["background_tasks"]["bounded"])
            self.assertEqual(profile["overall_level"], "L3")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("def materialize(items):\n    it = iter(items)\n    while True:\n        try:\n            next(it)\n        except StopIteration:\n            return\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertEqual(profile["background_tasks"]["count"], 0)

    def test_ffmpeg_not_ffprobe_raises_transcode_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("import subprocess\nsubprocess.run(['ffprobe', 'in.mp4'])\n", encoding="utf-8")
            probe = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertEqual(probe["scores"]["peak_cpu"], 1)
            (root / "main.py").write_text("import subprocess\nsubprocess.run(['ffmpeg', '-i', 'in.mp4', 'out.mp4'])\n", encoding="utf-8")
            transcode = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertEqual(transcode["scores"]["peak_cpu"], 3)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("import subprocess\nerror = subprocess.CalledProcessError(1, 'ffmpeg')\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertEqual(profile["external_processes"], [])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("from multiprocessing import Process\nworker = Process(target=run_worker)\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertIn("subprocess", profile["external_processes"])

    def test_download_and_small_reads_do_not_invent_cpu_or_memory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("def run(path):\n    config = path.read_text()\n    return download_file('https://example.test/a')\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertEqual(profile["scores"]["peak_cpu"], 0)
            self.assertEqual(profile["scores"]["peak_memory"], 0)
            self.assertEqual(profile["scores"]["network"], 3)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("async def run(response):\n    return await response.read()\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertEqual(profile["scores"]["peak_memory"], 2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("def generate_audio(response):\n    raw_audio_data = response.content\n    return raw_audio_data\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertEqual(profile["scores"]["peak_memory"], 2)

    def test_managed_local_model_subprocess_is_l4(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("from .gpt_sovits.adapter import run\n", encoding="utf-8")
            model_dir = root / "gpt_sovits"
            model_dir.mkdir()
            (model_dir / "adapter.py").write_text("import asyncio\nasync def run(command):\n    return await asyncio.create_subprocess_exec(*command)\n", encoding="utf-8")
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertIn("local_model", profile["features"])
            self.assertEqual(profile["overall_level"], "L4")

    def test_bounded_threaded_compaction_is_task_cpu_not_unbounded_loop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text(
                "import asyncio\n"
                "async def compact_history(items):\n"
                "    return await asyncio.to_thread(parse_items, items)\n",
                encoding="utf-8",
            )
            profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
            self.assertIn("cpu_task", profile["features"])
            self.assertEqual(profile["scores"]["peak_cpu"], 2)
            self.assertTrue(profile["background_tasks"]["bounded"])

    def test_runtime_index_promotes_source_profiles_and_keeps_metadata_fallback(self):
        from advisor.index import validate_index_semantics

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main.py").write_text("def noop():\n    return None\n", encoding="utf-8")
            source_profile = scanner.SourceAnalyzer(root, plugin_for(root)).profile()
        dimensions = ("idle_memory", "peak_memory", "idle_cpu", "peak_cpu", "disk", "network")
        fallback = {
            "plugin_id": "owner/fallback",
            "version": "1.0",
            "commit_sha": "b" * 40,
            "levels": {key: "L1" for key in dimensions},
            "scores": {key: 1 for key in dimensions},
            "features": [],
            "external_processes": [],
            "background_tasks": "unknown",
            "evidence": [],
            "unknowns": [],
            "confidence": 0.5,
            "evidence_level": "metadata_static",
            "scanned_at": "2026-08-24T00:00:00+00:00",
        }
        index = scanner.runtime_index(
            {"$meta": {"generated_at": "2026-08-25T00:00:00+00:00"}, "profiles": {"test/plugin": source_profile}},
            {"profiles": {"owner/fallback": fallback}},
        )
        validate_index_semantics(index, minimum_profiles=2)
        self.assertEqual(index["$meta"]["source_static_profile_count"], 1)
        self.assertEqual(index["$meta"]["metadata_fallback_profile_count"], 1)
        self.assertTrue(index["$meta"]["source_code_downloaded"])
        self.assertEqual(index["profiles"]["test/plugin"]["evidence_level"], "local_source_static_ast")
        unsafe = json.loads(json.dumps(index))
        unsafe["$meta"]["plugin_code_executed"] = True
        with self.assertRaises(ValueError):
            validate_index_semantics(unsafe, minimum_profiles=2)


if __name__ == "__main__":
    unittest.main()
