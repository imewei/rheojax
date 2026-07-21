[根目录](../../CLAUDE.md) > **transforms**

# transforms — 数据分析变换

## 模块职责

实现 11 个流变数据分析变换，每个变换继承 `rheojax.core.base.BaseTransform`，通过 `@TransformRegistry.register` 装饰器注册到全局 `Registry`。覆盖频域分析（FFT）、主曲线构建（TTS/SRFS）、LAOS 分析（OWChirp、SPP 分解）、粘弹性表征（Mutation Number、Cox-Merz、LVE Envelope）、谱恢复（Spectrum Inversion）、时频域转换（Prony）、信号处理（Smooth Derivative）。

## 入口与启动

- **即时导入**（与 `models` 相反，无懒加载）：`rheojax/transforms/__init__.py` 在包级直接 `from rheojax.transforms.<module> import <Class>`，导入本包即触发全部 `@TransformRegistry.register` 装饰器。
- `_ensure_all_registered()` 为 no-op 函数，仅为与 `rheojax.models._ensure_all_registered()` 保持调用对称性而存在（`PipelineBuilder._validate_components()` 两者都会调用）。

## 对外接口

```python
from rheojax.transforms import FFTAnalysis, Mastercurve, SRFS, SPPDecomposer, spp_analyze
result = FFTAnalysis().transform(data)
```

- **`FFTAnalysis`**（`fft_analysis.py`）— FFT 频谱分析。
- **`Mastercurve`**（`mastercurve.py`）— 时间-温度叠加（TTS）主曲线构建。
- **`SRFS`** + `detect_shear_banding` + `compute_shear_band_coexistence`（`srfs.py`）— 应变率-频率叠加（TTS 的应变率类比），含剪切带检测。
- **`MutationNumber`**（`mutation_number.py`）— 粘弹性特征数（0=纯弹性，1=纯粘性）。
- **`OWChirp`**（`owchirp.py`）— OWChirp 变换，用于 LAOS 分析。
- **`SmoothDerivative`**（`smooth_derivative.py`）— 平滑抗噪求导。
- **`SPPDecomposer`** + `spp_analyze`（`spp_decomposer.py`）— Sequence of Physical Processes（SPP）分解，用于 LAOS 分析；`spp_analyze` 为函数式便捷入口。
- **`PronyConversion`**（`prony_conversion.py`）— 经 Prony 级数实现时域↔频域转换。
- **`CoxMerz`**（`cox_merz.py`）— Cox-Merz 规则校验（复数粘度 |η*| vs 稳态粘度 η）。
- **`LVEEnvelope`**（`lve_envelope.py`）— 线性粘弹性起始应力包络。
- **`SpectrumInversion`**（`spectrum_inversion.py`）— 松弛谱 H(τ) 反演恢复。

## 关键依赖与配置

- 依赖 `rheojax.core`（`BaseTransform`、`TransformRegistry`、`RheoData`）。
- 依赖 `rheojax.utils.prony`（`PronyConversion`）、`rheojax.utils.spp_kernels`/`structure_factor`（`SPPDecomposer`）等专属数值核。
- 与 `models` 不同：本模块所有变换在包导入时立即注册，无 `_LAZY_IMPORTS` 表；新增变换只需在 `__init__.py` 顶层加一行 `import` 并追加 `__all__`。
- ruff 配置：`rheojax/transforms/*.py` 在 `pyproject.toml` 中豁免 E402（`safe_import_jax()` 可能需先于其他 import）。

## 数据模型

- 每个变换的 `.transform(data)` 接收/返回 `RheoData`（或其派生结果，如 `Mastercurve` 返回带平移因子的合并曲线，`SpectrumInversion` 返回松弛谱容器）。
- 无独立持久化状态；变换实例通常无状态或仅持有配置参数。

## 测试与质量

- `tests/transforms/`（13 个文件，镜像每个变换）：`test_fft_analysis`, `test_mastercurve`(+`_autoshift`), `test_srfs`, `test_mutation_number`, `test_owchirp`, `test_smooth_derivative`, `test_spp_decomposer`, `test_prony_conversion`, `test_cox_merz`, `test_lve_envelope`, `test_spectrum_inversion`, `test_batch_transform`（跨变换批处理集成）。

## 常见问题 (FAQ)

- **新增变换如何接入？** 在本目录新建文件，实现类继承 `BaseTransform`，加 `@TransformRegistry.register(name=...)`，然后在 `__init__.py` 顶层直接 `import` 该类并加入 `__all__`——**不要**加入懒加载表（与 `models` 约定不同）。
- **为什么 transforms 不像 models 那样懒加载？** 11 个变换的导入开销远小于 53 个模型（无 scipy/numpyro 级别的重依赖），即时导入换取更简单的注册模型（无需 `_ensure_all_registered()` 真实逻辑）。

## 相关文件清单

`cox_merz.py`, `fft_analysis.py`, `lve_envelope.py`, `mastercurve.py`, `mutation_number.py`, `owchirp.py`, `prony_conversion.py`, `smooth_derivative.py`, `spectrum_inversion.py`, `spp_decomposer.py`, `srfs.py`, `__init__.py`

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（补扫，此前根架构总览中标注为"未生成独立文档"）。
