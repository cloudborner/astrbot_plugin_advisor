import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class _Filter:
    class PermissionType:
        ADMIN = "admin"

    class EventMessageType:
        GROUP_MESSAGE = "group"

    @staticmethod
    def command(*_args, **_kwargs):
        return lambda function: function

    @staticmethod
    def permission_type(*_args, **_kwargs):
        return lambda function: function

    @staticmethod
    def event_message_type(*_args, **_kwargs):
        return lambda function: function


class PluginImportTests(unittest.TestCase):
    def test_main_imports_against_public_astrbot_api_shape(self):
        package = types.ModuleType("astrbot_plugin_advisor")
        package.__path__ = [str(ROOT)]
        astrbot = types.ModuleType("astrbot")
        astrbot.__version__ = "4.26.7"
        api = types.ModuleType("astrbot.api")
        api.AstrBotConfig = dict
        api.logger = types.SimpleNamespace(
            info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None
        )
        event = types.ModuleType("astrbot.api.event")
        event.AstrMessageEvent = object
        event.filter = _Filter
        message_components = types.ModuleType("astrbot.api.message_components")
        message_components.File = type("File", (), {})
        message_components.Plain = type("Plain", (), {})
        star = types.ModuleType("astrbot.api.star")
        star.Context = object
        star.Star = object
        star.StarTools = types.SimpleNamespace(
            get_data_dir=lambda _name: ROOT / ".test-data"
        )
        core = types.ModuleType("astrbot.core")
        core_star = types.ModuleType("astrbot.core.star")
        core_filter = types.ModuleType("astrbot.core.star.filter")
        command = types.ModuleType("astrbot.core.star.filter.command")
        command.GreedyStr = str
        stubs = {
            "astrbot_plugin_advisor": package,
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.message_components": message_components,
            "astrbot.api.star": star,
            "astrbot.core": core,
            "astrbot.core.star": core_star,
            "astrbot.core.star.filter": core_filter,
            "astrbot.core.star.filter.command": command,
        }
        with patch.dict(sys.modules, stubs, clear=False):
            spec = importlib.util.spec_from_file_location(
                "astrbot_plugin_advisor.main", ROOT / "main.py"
            )
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
                self.assertEqual(module.PLUGIN_NAME, "astrbot_plugin_advisor")
                self.assertTrue(hasattr(module.PluginAdvisor, "resource_profile"))
            finally:
                sys.modules.pop(spec.name, None)


if __name__ == "__main__":
    unittest.main()
