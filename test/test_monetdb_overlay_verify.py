#!/usr/bin/env python3
import importlib.util
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "monetdb_cira_compile", REPO_ROOT / "scripts" / "monetdb_cira_compile.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MonetdbOverlayVerifyTest(unittest.TestCase):
    def test_runtime_link_configuration_is_explicit_and_fail_closed(self):
        self.assertTrue(
            hasattr(MODULE, "runtime_link_args"),
            "missing MonetDB CIRA runtime link configuration",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            runtime = root / "build/runtime/libcira_runtime.so.1.0.0"
            runtime.parent.mkdir(parents=True)
            runtime.write_bytes(b"runtime")

            self.assertEqual(
                MODULE.runtime_link_args(root),
                [f"-DCIRA_RUNTIME_LIBRARY={runtime}"],
            )

            runtime.unlink()
            with self.assertRaisesRegex(RuntimeError, "CIRA runtime library"):
                MODULE.runtime_link_args(root)

    def test_host_module_split_rejects_retained_device_code(self):
        self.assertTrue(
            hasattr(MODULE, "verify_host_module_split"),
            "missing host/device LLVM split verifier",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            module = pathlib.Path(tempdir) / "host.ll"
            module.write_text(
                "define void @__cira_rv64_deadbeef(i64 %a, i64 %b) {\n"
                "  call void @cira_device_install_cacheline(i64 0)\n"
                "  ret void\n}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "retains RV64 device code"):
                MODULE.verify_host_module_split(module)

    def test_host_module_split_allows_unused_device_declarations(self):
        self.assertTrue(
            hasattr(MODULE, "verify_host_module_split"),
            "missing host/device LLVM split verifier",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            module = pathlib.Path(tempdir) / "host.ll"
            module.write_text(
                "declare void @cira_device_install_cacheline(i64)\n"
                "define void @host() { ret void }\n",
                encoding="utf-8",
            )
            MODULE.verify_host_module_split(module)

    def test_device_definition_removal_preserves_host_linkage(self):
        self.assertTrue(
            hasattr(MODULE, "remove_device_definitions"),
            "missing linkage-preserving host module split",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            combined = root / "combined.ll"
            host = root / "host.ll"
            combined.write_text(
                "@message = private constant [2 x i8] c\"x\\00\"\n"
                "define internal void @helper() {\n  ret void\n}\n"
                "define void @__cira_rv64_0123456789abcdef(i64 %a, i64 %b) {\n"
                "  call void @cira_device_install_cacheline(i64 0)\n"
                "  ret void\n}\n",
                encoding="utf-8",
            )

            MODULE.remove_device_definitions(
                combined, host, ["__cira_rv64_0123456789abcdef"]
            )

            text = host.read_text(encoding="utf-8")
            self.assertIn("@message = private constant", text)
            self.assertIn("define internal void @helper", text)
            self.assertNotIn("define void @__cira_rv64_", text)
            MODULE.verify_host_module_split(host)

    def test_finds_unique_real_versioned_library(self):
        self.assertTrue(
            hasattr(MODULE, "find_real_library"),
            "missing versioned library resolver",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            real = root / "libbat-11.52.0.so.29.0.0"
            real.write_bytes(b"library")
            (root / "libbat-11.52.0.so.29").symlink_to(real.name)
            (root / "libbat-11.52.0.so").symlink_to(real.name)

            self.assertEqual(MODULE.find_real_library(root, "libbat"), real)

    def test_object_overlay_preserves_ninja_dependency_mtime(self):
        self.assertTrue(
            hasattr(MODULE, "replace_object_preserving_ninja_mtime"),
            "missing Ninja-safe object replacement",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            source = root / "cira.o"
            destination = root / "native.o"
            source.write_bytes(b"cira-object")
            destination.write_bytes(b"native-object")
            expected_mtime = 1_700_000_000_123_456_789
            destination.touch()
            destination_stat = destination.stat()
            destination.chmod(destination_stat.st_mode)
            import os

            os.utime(destination, ns=(destination_stat.st_atime_ns, expected_mtime))

            MODULE.replace_object_preserving_ninja_mtime(source, destination)

            self.assertEqual(destination.read_bytes(), b"cira-object")
            self.assertEqual(destination.stat().st_mtime_ns, expected_mtime)

    def test_rejects_native_recompile_after_cira_overlay(self):
        self.assertTrue(
            hasattr(MODULE, "verify_final_overlays"),
            "missing final overlay verifier",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            cira_object = root / "cira.o"
            final_object = root / "final.o"
            cira_object.write_bytes(b"cira-object")
            final_object.write_bytes(b"native-object")
            records = {
                "gdk/gdk_aggr.c": {"artifacts": {"host_object": str(cira_object)}}
            }
            final_objects = {"gdk/gdk_aggr.c": final_object}

            with self.assertRaisesRegex(RuntimeError, "overwritten after CIRA overlay"):
                MODULE.verify_final_overlays(records, final_objects)

    def test_accepts_byte_identical_final_overlay(self):
        self.assertTrue(
            hasattr(MODULE, "verify_final_overlays"),
            "missing final overlay verifier",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            cira_object = root / "cira.o"
            final_object = root / "final.o"
            cira_object.write_bytes(b"same-object")
            final_object.write_bytes(b"same-object")
            records = {
                "gdk/gdk_aggr.c": {"artifacts": {"host_object": str(cira_object)}}
            }
            final_objects = {"gdk/gdk_aggr.c": final_object}

            MODULE.verify_final_overlays(records, final_objects)


if __name__ == "__main__":
    unittest.main()
