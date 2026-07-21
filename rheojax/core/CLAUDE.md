[根目录](../../CLAUDE.md) > **core**

# core — 抽象基类与拟合编排

## 模块职责

`rheojax.core` 是全仓的地基模块：定义 `BaseModel`/`BaseTransform` 抽象接口、`RheoData` 数据容器、参数系统（`Parameter`/`ParameterSet`）、插件注册表（`Registry`/`ModelRegistry`/`TransformRegistry`）、NLSQ↔贝叶斯拟合编排（`FitOrchestrator`）、NumPyro 模型构建与 ArviZ 诊断封装。`rheojax/models/*`、`rheojax/transforms/*`、`rheojax/pipeline/*`、`rheojax/gui/*`、`rheojax/cli/*` 均依赖本模块。

## 入口与启动

无独立可执行入口；作为库被导入。关键初始化点：
- `rheojax/core/__init__.py` — 导出公共 API，初始化全局单例 `Registry.get_instance()`。
- `rheojax/core/jax_config.py` — `safe_import_jax()`：强制 float64 的 JAX 导入入口，全仓唯一合法 JAX 导入路径；`lazy_import()` 用于延迟导入重依赖（如 diffrax）。

## 对外接口

- **基类**：`BaseModel`（继承 `BayesianMixin` + `ABC`，子类必须实现 `_fit()`/`_predict()`）、`BaseTransform`、`TransformPipeline`。
- **数据**：`RheoData`（流变数据容器，`rheojax/core/data.py`）。
- **结果**：`FitResult`、`ModelInfo`、`ModelComparison`（`fit_result.py`）；`BayesianResult`（`bayesian_result.py`）。
- **参数**：`Parameter`、`ParameterSet`、`ParameterConstraint`、`SharedParameterSet`、`ParameterOptimizer`（`parameters.py`）。
- **注册表**：`Registry`、`ModelRegistry`、`TransformRegistry`、`PluginInfo`、`PluginType`（枚举 MODEL/TRANSFORM）（`registry.py`）。线程安全（`_discover_lock`）。
- **测试模式/几何**：`TestMode`、`detect_test_mode()`、`validate_test_mode()`（`test_modes.py`）。
- **贝叶斯**：`BayesianMixin`（提供 `fit_bayesian()`、`sample_prior()`、`get_credible_intervals()`）、`bayesian_diagnostics.py`、`numpyro_model_builder.py`、`arviz_utils.py`（arviz 1.x kwarg 转换 shim）。
- **编排**：`FitOrchestrator`（`fit_orchestrator.py`）— NLSQ→NUTS 热启动流程的核心协调器。
- **校验**：`_validation.py`（`reject_removed_options`）、`post_fit_validator.py`。
- **物料清单**：`inventory.py`（`Protocol`、`TransformType` 枚举，供模型/变换声明支持的测试协议）。

## 关键依赖与配置

- 依赖 `nlsq`（必须先于 JAX 导入）、`jax`/`jaxlib`（float64 强制）、`numpyro`、`arviz` 1.x 系列（`arviz-base`/`arviz-stats`/`arviz-plots`）。
- `rheojax/logging` 提供 `get_logger()` 供全模块结构化日志。
- 无独立配置文件；行为通过环境变量（如 `RHEOJAX_*`，见 `parallel` 模块）间接影响拟合编排。

## 数据模型

- `RheoData`：承载时间/频率域流变测量（应力、应变、模量等）及元数据（测试模式、单位）。
- `ParameterSet`：命名参数集合，含边界约束（`ParameterConstraint`）与共享参数（`SharedParameterSet`，用于多模式模型如 GeneralizedMaxwell）。
- `FitResult`：NLSQ 拟合结果（参数值、协方差、R²等）；`BayesianResult`：NUTS 采样结果（InferenceData 封装）。
- `PluginInfo`：`{name, plugin_class, plugin_type}`，供 `Registry.inventory()` 枚举全部已注册模型/变换。

## 测试与质量

- 对应测试目录：`tests/core/`（约 20 个文件），覆盖 `base`（含 edge cases）、`bayesian`（含 internals/mode_closure/edge_cases）、`registry`（含一致性检查）、`parameters`、`data`、`fit_result`、`fit_orchestrator`、`jax_config`、`float64_precision`、`numpyro_model_builder`、`arviz_utils`、`test_modes`。
- **已知缺陷（未修复，故意保留）**：`tests/core/test_base.py` 中存在重复的 `TestBaseModel` 类定义，第二个类静默遮蔽第一个，约 600 行测试从未被 pytest 收集执行。修改此文件前务必核实类名唯一性。

## 常见问题 (FAQ)

- **为什么不能直接 `import jax`？** 会绕过 `jax.config.update("jax_enable_x64", True)` 的强制设置，导致 float32 静默回退（NLSQ ≥0.2.1 默认行为）。统一走 `safe_import_jax()`。
- **模型注册何时生效？** `@ModelRegistry.register` 装饰器在模型子模块被 import 时触发；由于 `rheojax.models` 使用懒加载，需要显式调用 `rheojax.models._ensure_all_registered()` 才能让 `Registry.inventory()` 看到全部 53 个模型。

## 相关文件清单

`base.py`, `data.py`, `fit_result.py`, `parameters.py`, `registry.py`, `test_modes.py`, `jax_config.py`, `bayesian.py`, `bayesian_result.py`, `bayesian_diagnostics.py`, `numpyro_model_builder.py`, `arviz_utils.py`, `fit_orchestrator.py`, `inventory.py`, `post_fit_validator.py`, `_validation.py`, `__init__.py`

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（架构扫描）。
