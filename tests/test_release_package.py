import hashlib
import importlib.util
import tempfile
import zipfile
from pathlib import Path

from scripts.build_release import PACKAGE_ROOT, ROOT, build_release


def test_release_archive_has_one_valid_plugin_root_and_no_staging_content():
    with tempfile.TemporaryDirectory() as directory:
        archive, checksum = build_release(Path(directory))
        assert archive.exists()
        assert checksum.exists()
        expected = checksum.read_text(encoding="ascii").split()[0]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            assert names
            assert all(name.startswith(f"{PACKAGE_ROOT}/") for name in names)
            assert f"{PACKAGE_ROOT}/main.py" in names
            assert f"{PACKAGE_ROOT}/metadata.yaml" in names
            assert f"{PACKAGE_ROOT}/data/plugin_capabilities.json" in names
            assert f"{PACKAGE_ROOT}/data/source_resource_index.json" in names
            assert f"{PACKAGE_ROOT}/data/market_snapshot.json" in names
            assert not any(name.startswith(f"{PACKAGE_ROOT}/schemas/") for name in names)
            for excluded in (
                "advisor/remote_index.py",
                "data/index_public_key.pem",
                "data/resource_profiles.json",
                "data/resource_profiles.manifest.json",
                "data/source_resource_profiles.json",
                "data/source_resource_review_queue.json",
                "data/source_function_evidence.json",
                "data/source_function_llm_profiles.json",
                "data/source_function_llm_profiles_v2.json",
                "data/source_function_llm_profiles_v3.json",
                "data/source_function_llm_profiles_v3_reviewed.json",
            ):
                assert f"{PACKAGE_ROOT}/{excluded}" not in names
            runtime_requirements = bundle.read(
                f"{PACKAGE_ROOT}/requirements.txt"
            ).decode("utf-8")
            assert "cryptography" not in runtime_requirements.casefold()
            assert not any("/.advisor-upload-" in name for name in names)
            assert not any("/__pycache__/" in name for name in names)
            assert not any(name.startswith(f"{PACKAGE_ROOT}/tests/") for name in names)
            assert f"{PACKAGE_ROOT}/pytest.ini" not in names
            assert not any(name.startswith(f"{PACKAGE_ROOT}/artifacts/") for name in names)
            assert not any(name.startswith(f"{PACKAGE_ROOT}/build/") for name in names)
            assert not any(name.startswith(f"{PACKAGE_ROOT}/dist/") for name in names)
            assert not any(
                name.startswith(f"{PACKAGE_ROOT}/source_archives/") for name in names
            )
            assert not any(
                name.startswith(f"{PACKAGE_ROOT}/source_extracted/") for name in names
            )


def test_release_builder_refuses_output_inside_plugin_directory():
    try:
        build_release(ROOT / "dist")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("output inside the plugin tree must be rejected")


def test_extracted_release_keeps_python_package_name_stable():
    with tempfile.TemporaryDirectory() as output, tempfile.TemporaryDirectory() as extracted:
        archive, _checksum = build_release(Path(output))
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        package = Path(extracted, PACKAGE_ROOT)
        assert package.is_dir()
        spec = importlib.util.spec_from_file_location(
            PACKAGE_ROOT,
            package / "__init__.py",
            submodule_search_locations=[str(package)],
        )
        assert spec is not None
        assert spec.name == "astrbot_plugin_advisor"
