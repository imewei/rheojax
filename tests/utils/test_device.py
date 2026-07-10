"""Tests for rheojax.utils.device (GPU detection utilities).

The real environment here has nvcc/nvidia-smi present but CPU-only jaxlib, so
the smoke tests below exercise whatever the machine reports. Every branch-level
test mocks ``subprocess.run``, the module-level helpers, and ``safe_import_jax``
so the GPU-available and CPU-only paths run deterministically regardless of
hardware.
"""

import subprocess
import types
from unittest import mock

import pytest

from rheojax.utils import device
from rheojax.utils.device import (
    check_gpu_availability,
    check_plugin_conflicts,
    get_device_info,
    get_gpu_info,
    get_gpu_memory_info,
    get_recommended_package,
    get_system_cuda_version,
    print_device_summary,
)


class _FakeProc:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def _fake_jax(devices, backend="cpu", version="0.8.0"):
    """Build a minimal fake jax module exposing the API device.py uses."""
    return types.SimpleNamespace(
        __version__=version,
        default_backend=lambda: backend,
        devices=lambda: list(devices),
    )


@pytest.mark.smoke
class TestDeviceUtilities:
    """Tests for GPU detection utilities."""

    def test_get_system_cuda_version_returns_tuple(self):
        """Test that function returns a 2-tuple regardless of GPU presence."""
        result = get_system_cuda_version()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_system_cuda_version_types(self):
        version, major = get_system_cuda_version()
        # Either both None or both populated
        if version is not None:
            assert isinstance(version, str)
            assert isinstance(major, int)
        else:
            assert major is None

    def test_get_gpu_info_returns_tuple(self):
        result = get_gpu_info()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_gpu_info_types(self):
        name, sm = get_gpu_info()
        if name is not None:
            assert isinstance(name, str)
            assert isinstance(sm, float)
        else:
            assert sm is None


class TestGetSystemCudaVersion:
    """Branch coverage for get_system_cuda_version via mocked nvcc."""

    def test_parses_release_line(self):
        out = "nvcc: NVIDIA (R) Cuda compiler\nCuda compilation tools, release 12.6, V12.6.77\n"
        with mock.patch.object(device.subprocess, "run", return_value=_FakeProc(0, out)):
            version, major = get_system_cuda_version()
        assert version == "12.6"
        assert major == 12

    def test_nonzero_returncode_returns_none(self):
        with mock.patch.object(device.subprocess, "run", return_value=_FakeProc(1, "")):
            assert get_system_cuda_version() == (None, None)

    def test_timeout_returns_none(self):
        with mock.patch.object(
            device.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvcc", 5)
        ):
            assert get_system_cuda_version() == (None, None)

    def test_not_found_returns_none(self):
        with mock.patch.object(device.subprocess, "run", side_effect=FileNotFoundError):
            assert get_system_cuda_version() == (None, None)

    def test_unparseable_version_returns_none(self):
        # "release X.Y" -> int("X") raises ValueError, caught and logged.
        out = "Cuda compilation tools, release X.Y, Vbogus\n"
        with mock.patch.object(device.subprocess, "run", return_value=_FakeProc(0, out)):
            assert get_system_cuda_version() == (None, None)

    def test_generic_exception_returns_none(self):
        with mock.patch.object(device.subprocess, "run", side_effect=RuntimeError("boom")):
            assert get_system_cuda_version() == (None, None)


class TestGetGpuInfo:
    """Branch coverage for get_gpu_info via mocked nvidia-smi."""

    def test_parses_name_and_compute_cap(self):
        with mock.patch.object(
            device.subprocess,
            "run",
            return_value=_FakeProc(0, "NVIDIA GeForce RTX 4090, 8.9\n"),
        ):
            name, sm = get_gpu_info()
        assert name == "NVIDIA GeForce RTX 4090"
        assert sm == pytest.approx(8.9)

    def test_empty_stdout_returns_none(self):
        with mock.patch.object(device.subprocess, "run", return_value=_FakeProc(0, "  \n")):
            assert get_gpu_info() == (None, None)

    def test_timeout_returns_none(self):
        with mock.patch.object(
            device.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)
        ):
            assert get_gpu_info() == (None, None)

    def test_not_found_returns_none(self):
        with mock.patch.object(device.subprocess, "run", side_effect=FileNotFoundError):
            assert get_gpu_info() == (None, None)

    def test_unparseable_compute_cap_returns_none(self):
        with mock.patch.object(
            device.subprocess, "run", return_value=_FakeProc(0, "SomeGPU, notafloat\n")
        ):
            assert get_gpu_info() == (None, None)

    def test_generic_exception_returns_none(self):
        with mock.patch.object(device.subprocess, "run", side_effect=RuntimeError("boom")):
            assert get_gpu_info() == (None, None)


class TestGetRecommendedPackage:
    """Every compatibility branch of get_recommended_package."""

    def _patch(self, cuda, gpu):
        return mock.patch.multiple(
            device,
            get_system_cuda_version=mock.Mock(return_value=cuda),
            get_gpu_info=mock.Mock(return_value=gpu),
        )

    def test_no_cuda_returns_none(self):
        with self._patch((None, None), ("GPU", 8.9)):
            assert get_recommended_package() is None

    def test_no_gpu_returns_none(self):
        with self._patch(("12.6", 12), (None, None)):
            assert get_recommended_package() is None

    def test_cuda13_modern_gpu(self):
        with self._patch(("13.0", 13), ("GPU", 8.9)):
            assert get_recommended_package() == "jax[cuda13-local]"

    def test_cuda13_old_gpu_returns_none(self):
        with self._patch(("13.0", 13), ("GPU", 7.0)):
            assert get_recommended_package() is None

    def test_cuda12_modern_gpu(self):
        with self._patch(("12.6", 12), ("GPU", 6.1)):
            assert get_recommended_package() == "jax[cuda12-local]"

    def test_cuda12_old_gpu_returns_none(self):
        with self._patch(("12.6", 12), ("GPU", 3.5)):
            assert get_recommended_package() is None

    def test_unsupported_cuda_major_returns_none(self):
        with self._patch(("11.8", 11), ("GPU", 8.9)):
            assert get_recommended_package() is None


class TestCheckPluginConflicts:
    """Plugin-conflict detection with mocked importlib.metadata."""

    def _patch_versions(self, versions: dict):
        import importlib.metadata as md

        def fake_version(name):
            if name in versions:
                return versions[name]
            raise md.PackageNotFoundError(name)

        return mock.patch.object(md, "version", side_effect=fake_version)

    def test_no_plugins_returns_empty(self):
        with self._patch_versions({"jaxlib": "0.8.0"}):
            assert check_plugin_conflicts() == []

    def test_dual_plugin_conflict(self):
        versions = {
            "jaxlib": "0.8.0",
            "jax-cuda12-plugin": "0.8.0",
            "jax-cuda13-plugin": "0.8.0",
        }
        with self._patch_versions(versions):
            issues = check_plugin_conflicts()
        assert any("Both cuda12" in i for i in issues)

    def test_version_mismatch(self):
        versions = {"jaxlib": "0.8.0", "jax-cuda12-plugin": "0.7.0"}
        with self._patch_versions(versions):
            issues = check_plugin_conflicts()
        assert any("must exactly match jaxlib" in i for i in issues)

    def test_exception_returns_empty(self):
        import importlib.metadata as md

        with mock.patch.object(md, "version", side_effect=RuntimeError("boom")):
            assert check_plugin_conflicts() == []


class TestCheckGpuAvailability:
    """check_gpu_availability across GPU-present / CPU-only / error paths."""

    def test_no_gpu_hardware_returns_false(self):
        with mock.patch.object(device, "get_gpu_info", return_value=(None, None)):
            assert check_gpu_availability() is False

    def test_gpu_used_by_jax_returns_true(self):
        fake = _fake_jax(["cuda:0"], backend="gpu")
        with (
            mock.patch.object(device, "get_gpu_info", return_value=("GPU", 8.9)),
            mock.patch.object(device, "get_system_cuda_version", return_value=("12.6", 12)),
            mock.patch.object(device, "check_plugin_conflicts", return_value=["some issue"]),
            mock.patch(
                "rheojax.core.jax_config.safe_import_jax", return_value=(fake, None)
            ),
        ):
            assert check_gpu_availability() is True

    def test_gpu_present_but_cpu_backend_warns_and_returns_false(self, capsys):
        fake = _fake_jax(["cpu:0"], backend="cpu")
        with (
            mock.patch.object(device, "get_gpu_info", return_value=("GPU", 8.9)),
            mock.patch.object(device, "get_system_cuda_version", return_value=("12.6", 12)),
            mock.patch.object(device, "check_plugin_conflicts", return_value=[]),
            mock.patch.object(device, "get_recommended_package", return_value=None),
            mock.patch(
                "rheojax.core.jax_config.safe_import_jax", return_value=(fake, None)
            ),
        ):
            result = check_gpu_availability(warn=True)
        assert result is False
        assert "GPU AVAILABLE BUT NOT USED" in capsys.readouterr().out

    def test_gpu_present_cpu_backend_no_warn_is_silent(self, capsys):
        fake = _fake_jax(["cpu:0"], backend="cpu")
        with (
            mock.patch.object(device, "get_gpu_info", return_value=("GPU", 8.9)),
            mock.patch.object(device, "get_system_cuda_version", return_value=("12.6", 12)),
            mock.patch(
                "rheojax.core.jax_config.safe_import_jax", return_value=(fake, None)
            ),
        ):
            result = check_gpu_availability(warn=False)
        assert result is False
        assert capsys.readouterr().out == ""

    def test_jax_import_error_returns_false(self):
        with (
            mock.patch.object(device, "get_gpu_info", return_value=("GPU", 8.9)),
            mock.patch.object(device, "get_system_cuda_version", return_value=("12.6", 12)),
            mock.patch(
                "rheojax.core.jax_config.safe_import_jax", side_effect=ImportError
            ),
        ):
            assert check_gpu_availability() is False

    def test_generic_exception_returns_false(self):
        with (
            mock.patch.object(device, "get_gpu_info", return_value=("GPU", 8.9)),
            mock.patch.object(device, "get_system_cuda_version", return_value=("12.6", 12)),
            mock.patch(
                "rheojax.core.jax_config.safe_import_jax", side_effect=RuntimeError("boom")
            ),
        ):
            assert check_gpu_availability() is False


class TestPrintGpuWarning:
    """_print_gpu_warning output branches."""

    def test_prints_issues_and_package(self, capsys):
        fake = _fake_jax(["cpu:0"], backend="cpu")
        with (
            mock.patch.object(device, "check_plugin_conflicts", return_value=["conflict A"]),
            mock.patch.object(
                device, "get_recommended_package", return_value="jax[cuda12-local]"
            ),
            mock.patch(
                "rheojax.core.jax_config.safe_import_jax", return_value=(fake, None)
            ),
        ):
            device._print_gpu_warning("GPU", 8.9, "12.6", 12)
        out = capsys.readouterr().out
        assert "Issues detected" in out
        assert "conflict A" in out
        assert "jax[cuda12-local]" in out

    def test_handles_jax_import_error(self, capsys):
        with (
            mock.patch.object(device, "check_plugin_conflicts", return_value=[]),
            mock.patch.object(device, "get_recommended_package", return_value=None),
            mock.patch(
                "rheojax.core.jax_config.safe_import_jax", side_effect=ImportError
            ),
        ):
            device._print_gpu_warning("GPU", 8.9, None, None)
        assert "JAX backend: unknown" in capsys.readouterr().out


class TestGetDeviceInfo:
    """get_device_info aggregation and ImportError fallback."""

    def test_populates_from_fake_jax(self):
        fake = _fake_jax(["cuda:0", "cuda:1"], backend="gpu", version="0.8.0")
        with (
            mock.patch("rheojax.core.jax_config.safe_import_jax", return_value=(fake, None)),
            mock.patch.object(device, "get_gpu_info", return_value=("GPU", 8.9)),
            mock.patch.object(device, "get_system_cuda_version", return_value=("12.6", 12)),
            mock.patch.object(
                device, "get_recommended_package", return_value="jax[cuda12-local]"
            ),
            mock.patch.object(device, "check_plugin_conflicts", return_value=[]),
        ):
            info = get_device_info()
        assert info["jax_version"] == "0.8.0"
        assert info["jax_backend"] == "gpu"
        assert info["gpu_count"] == 2
        assert info["using_gpu"] is True
        assert info["gpu_hardware"] == "GPU"
        assert info["system_cuda_major"] == 12
        assert info["recommended_package"] == "jax[cuda12-local]"

    def test_jax_import_error_leaves_jax_fields_none(self):
        with (
            mock.patch("rheojax.core.jax_config.safe_import_jax", side_effect=ImportError),
            mock.patch.object(device, "get_gpu_info", return_value=(None, None)),
            mock.patch.object(device, "get_system_cuda_version", return_value=(None, None)),
            mock.patch.object(device, "get_recommended_package", return_value=None),
            mock.patch.object(device, "check_plugin_conflicts", return_value=[]),
        ):
            info = get_device_info()
        assert info["jax_version"] is None
        assert info["jax_backend"] is None
        assert info["devices"] == []
        assert info["using_gpu"] is False


class TestGetGpuMemoryInfo:
    """get_gpu_memory_info parsing branches."""

    def test_parses_memory_fields(self):
        with mock.patch.object(
            device.subprocess, "run", return_value=_FakeProc(0, "16384, 2048, 14336, 25\n")
        ):
            info = get_gpu_memory_info()
        assert info == {
            "total_mb": 16384,
            "used_mb": 2048,
            "free_mb": 14336,
            "utilization_percent": 25,
        }

    def test_nonzero_returncode_returns_empty(self):
        with mock.patch.object(device.subprocess, "run", return_value=_FakeProc(1, "")):
            assert get_gpu_memory_info() == {}

    def test_not_found_returns_empty(self):
        with mock.patch.object(device.subprocess, "run", side_effect=FileNotFoundError):
            assert get_gpu_memory_info() == {}

    def test_unparseable_values_returns_empty(self):
        with mock.patch.object(
            device.subprocess, "run", return_value=_FakeProc(0, "a, b, c, d\n")
        ):
            assert get_gpu_memory_info() == {}


class TestPrintDeviceSummary:
    """print_device_summary GPU / CPU / no-JAX branches."""

    def test_gpu_branch_reports_memory(self, capsys):
        fake = _fake_jax(["cuda:0"], backend="gpu")
        with (
            mock.patch("rheojax.core.jax_config.safe_import_jax", return_value=(fake, None)),
            mock.patch.object(
                device,
                "get_gpu_memory_info",
                return_value={
                    "total_mb": 16384,
                    "used_mb": 2048,
                    "free_mb": 14336,
                    "utilization_percent": 25,
                },
            ),
        ):
            print_device_summary()
        out = capsys.readouterr().out
        assert "Using: GPU acceleration" in out
        assert "2048/16384 MB" in out

    def test_cpu_branch_calls_availability_check(self, capsys):
        fake = _fake_jax(["cpu:0"], backend="cpu")
        with (
            mock.patch("rheojax.core.jax_config.safe_import_jax", return_value=(fake, None)),
            mock.patch.object(device, "check_gpu_availability", return_value=False) as chk,
        ):
            print_device_summary()
        assert "Using: CPU-only" in capsys.readouterr().out
        chk.assert_called_once()

    def test_no_jax_reports_not_installed(self, capsys):
        with mock.patch(
            "rheojax.core.jax_config.safe_import_jax", side_effect=ImportError
        ):
            print_device_summary()
        assert "JAX not installed" in capsys.readouterr().out
