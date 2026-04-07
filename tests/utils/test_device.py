"""Tests for rheojax.utils.device (GPU detection utilities)."""

import pytest

from rheojax.utils.device import get_gpu_info, get_system_cuda_version


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
