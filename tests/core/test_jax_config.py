"""Tests for safe_import_jax() and float64 configuration."""

import pytest


@pytest.mark.smoke
class TestSafeImportJax:
    """Tests for safe_import_jax()."""

    def test_returns_jax_module(self):
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        assert jax is not None
        assert jnp is not None

    def test_jax_has_expected_attrs(self):
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        assert hasattr(jax, "jit")
        assert hasattr(jax, "grad")
        assert hasattr(jnp, "array")
        assert hasattr(jnp, "float64")

    def test_float64_enabled(self):
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        test_array = jnp.array([1.0])
        assert test_array.dtype == jnp.float64

    def test_verify_float64_passes(self):
        from rheojax.core.jax_config import safe_import_jax, verify_float64

        safe_import_jax()
        # Should not raise
        verify_float64()

    def test_idempotent(self):
        """Calling safe_import_jax() multiple times returns same modules."""
        from rheojax.core.jax_config import safe_import_jax

        jax1, jnp1 = safe_import_jax()
        jax2, jnp2 = safe_import_jax()
        assert jax1 is jax2
        assert jnp1 is jnp2

    def test_verify_float64_failure_is_wrapped_in_runtime_error(self, monkeypatch):
        """The RuntimeError from verify_float64() must be caught and re-raised
        with additional context, not left to propagate unwrapped."""
        from rheojax.core import jax_config

        jax_config.reset_validation()

        def boom():
            raise RuntimeError("dtype mismatch")

        monkeypatch.setattr(jax_config, "verify_float64", boom)

        with pytest.raises(RuntimeError, match="Float64 verification failed"):
            jax_config.safe_import_jax()

        # Leave cached state clean so later tests re-run the (unpatched,
        # real) validation path once monkeypatch tears down.
        jax_config.reset_validation()


class TestEnableCompilationCache:
    """Tests for _enable_compilation_cache()."""

    def test_sets_min_compile_time_secs_below_default_threshold(self, monkeypatch, tmp_path):
        """jax_persistent_cache_min_compile_time_secs defaults to 1.0s in the
        installed JAX, which silently skips persisting sub-second rheology
        model compiles. It must be explicitly lowered."""
        import pathlib
        from unittest.mock import MagicMock

        from rheojax.core.jax_config import _enable_compilation_cache

        monkeypatch.delenv("RHEOJAX_NO_JIT_CACHE", raising=False)
        monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)

        fake_jax = MagicMock()
        _enable_compilation_cache(fake_jax)

        calls = dict(
            call.args for call in fake_jax.config.update.call_args_list
        )
        assert calls["jax_compilation_cache_dir"] == str(
            tmp_path / ".cache" / "rheojax" / "jax_cache"
        )
        assert "jax_persistent_cache_min_compile_time_secs" in calls
        assert calls["jax_persistent_cache_min_compile_time_secs"] < 1.0

    def test_disabled_by_env_var(self, monkeypatch):
        from unittest.mock import MagicMock

        from rheojax.core.jax_config import _enable_compilation_cache

        monkeypatch.setenv("RHEOJAX_NO_JIT_CACHE", "1")
        fake_jax = MagicMock()
        _enable_compilation_cache(fake_jax)
        fake_jax.config.update.assert_not_called()

    def test_falls_back_to_experimental_api_on_attribute_error(self, monkeypatch, tmp_path):
        """If the config-based API raises (e.g. older JAX), fall back to the
        experimental compilation_cache module instead of crashing."""
        import pathlib
        from unittest.mock import MagicMock

        from jax.experimental.compilation_cache import compilation_cache as cc

        from rheojax.core.jax_config import _enable_compilation_cache

        monkeypatch.delenv("RHEOJAX_NO_JIT_CACHE", raising=False)
        monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)

        fake_jax = MagicMock()
        fake_jax.config.update.side_effect = AttributeError("no such config option")

        set_cache_dir_mock = MagicMock()
        monkeypatch.setattr(cc, "set_cache_dir", set_cache_dir_mock)

        with pytest.warns(RuntimeWarning, match="min-compile-time"):
            _enable_compilation_cache(fake_jax)

        set_cache_dir_mock.assert_called_once_with(
            str(tmp_path / ".cache" / "rheojax" / "jax_cache")
        )

    def test_home_resolution_failure_is_non_fatal(self, monkeypatch):
        """pathlib.Path.home() raises RuntimeError when $HOME is unset and the
        UID has no /etc/passwd entry (e.g. arbitrary-UID containers). This
        must be swallowed, not propagate out of safe_import_jax()."""
        import pathlib
        from unittest.mock import MagicMock

        from rheojax.core.jax_config import _enable_compilation_cache

        monkeypatch.delenv("RHEOJAX_NO_JIT_CACHE", raising=False)

        def raise_runtime_error():
            raise RuntimeError("Could not determine home directory.")

        monkeypatch.setattr(pathlib.Path, "home", raise_runtime_error)

        fake_jax = MagicMock()
        _enable_compilation_cache(fake_jax)  # must not raise

        fake_jax.config.update.assert_not_called()


class TestLazyImport:
    """Tests for lazy_import() / _LazyModule."""

    def test_defers_import_until_first_attribute_access(self, monkeypatch):
        import importlib
        from unittest.mock import MagicMock

        from rheojax.core.jax_config import lazy_import

        fake_module = MagicMock()
        fake_module.some_attr = "value"
        import_mock = MagicMock(return_value=fake_module)
        monkeypatch.setattr(importlib, "import_module", import_mock)

        proxy = lazy_import("some.fake.module")
        import_mock.assert_not_called()

        assert proxy.some_attr == "value"
        import_mock.assert_called_once_with("some.fake.module")

    def test_forwards_calls_to_loaded_module(self, monkeypatch):
        import importlib
        from unittest.mock import MagicMock

        from rheojax.core.jax_config import lazy_import

        fake_module = MagicMock()
        fake_module.some_method.return_value = 42
        monkeypatch.setattr(importlib, "import_module", MagicMock(return_value=fake_module))

        proxy = lazy_import("some.fake.module")
        assert proxy.some_method() == 42

    def test_loads_module_only_once_across_accesses(self, monkeypatch):
        import importlib
        from unittest.mock import MagicMock

        from rheojax.core.jax_config import lazy_import

        fake_module = MagicMock()
        import_mock = MagicMock(return_value=fake_module)
        monkeypatch.setattr(importlib, "import_module", import_mock)

        proxy = lazy_import("some.fake.module")
        _ = proxy.attr_a
        _ = proxy.attr_b
        assert import_mock.call_count == 1

    def test_repr_reflects_loaded_state(self, monkeypatch):
        import importlib
        from unittest.mock import MagicMock

        from rheojax.core.jax_config import lazy_import

        fake_module = MagicMock()
        monkeypatch.setattr(importlib, "import_module", MagicMock(return_value=fake_module))

        proxy = lazy_import("some.fake.module")
        assert "loaded=False" in repr(proxy)

        _ = proxy.anything

        assert "loaded=True" in repr(proxy)
