[根目录](../../CLAUDE.md) > **utils**

# utils — 数值工具与优化辅助

## 模块职责

提供跨模型共享的数值计算工具：Mittag-Leffler 函数（分数阶模型核心）、Prony 级数处理、NLSQ 优化封装、物理合理性检查、不确定度量化（bootstrap/Hessian 置信区间）、GPU 设备检测、各模型族专属数值核（EPM/HL/SGR/MCT/SPP kernels）、模型初始参数估计启发式（`initialization/` 子包，覆盖全部分数阶模型族）。

## 入口与启动

无独立可执行入口；纯工具库。`rheojax/utils/__init__.py` 汇总最常用的公共 API（设备检测、指标、优化、物理检查、不确定度）；专项工具（如 `mittag_leffler.py`, `prony.py`, 各 `*_kernels.py`）需从子模块直接导入。

## 对外接口

- **优化**：`nlsq_optimize`, `optimize_with_bounds`, `optimize`, `fit_parameters`, `create_least_squares_objective`, `residual_sum_of_squares`, `OptimizationResult`（`optimization.py`）。
- **模型选择/结果构建**：`build_fit_result`, `compare_models`（`model_selection.py`）。
- **设备**：`check_gpu_availability`, `check_plugin_conflicts`, `get_device_info`, `get_gpu_memory_info`, `print_device_summary`（`device.py`）— CLI `info` 命令与 GUI 设备状态栏均调用。
- **指标**：`compute_fit_quality`, `r2_complex`（`metrics.py`，支持复数模量 R²）。
- **物理校验**：`PhysicsViolation`, `check_fit_physics`（`physics_checks.py`）。
- **不确定度**：`bootstrap_ci`, `hessian_ci`（`uncertainty.py`）。
- **分数阶数学**：`mittag_leffler.py`（Mittag-Leffler 函数，注意 `pyproject.toml` 中此文件豁免 E402 因 `safe_import_jax()` 需先于其他 import）。
- **Prony**：`prony.py`（`compute_r_squared`, `create_prony_parameter_set`, `select_optimal_n`, `softmax_penalty` — 被 `models/multimode/generalized_maxwell.py` 直接使用）。
- **初始化启发式**：`initialization/` 子包，每个分数阶模型族一个文件（`fractional_maxwell_model.py`, `fractional_kelvin_voigt.py`, `fractional_zener_*.py`, `fractional_burgers.py`, `fractional_jeffreys.py`, `fractional_poynting_thomson.py`）+ `auto_p0.py`（自动初始猜测）+ `base.py` + `constants.py`。
- **模型专属核**：`epm_kernels.py`(+`_tensorial`), `hl_kernels.py`, `sgr_kernels.py`, `sgr_monte_carlo.py`, `sgr_population_balance.py`, `mct_kernels.py`, `spp_kernels.py`, `structure_factor.py`, `volterra_creep.py`。
- **其他**：`data_quality.py`, `compatibility.py`, `protocol_preprocessing.py`。

## 关键依赖与配置

- `scipy`、`numpy`；JAX 相关工具通过 `safe_import_jax()` 获取（`rheojax.core.jax_config`）。
- `pyproject.toml` 中 `rheojax/utils/mittag_leffler.py` 与 `rheojax/utils/initialization/*.py` 均被列为 E402 豁免路径——修改这些文件时不要把 import 顺序"修正"为标准顺序。

## 数据模型

- 无独立持久化数据模型；主要操作 `ParameterSet`（core）与原始数组（NumPy/JAX）。

## 测试与质量

- `tests/utils/`（30 个文件，含 `initialization/` 子目录 2 个文件）：覆盖每个公共函数（`test_optimization`, `test_nlsq_optimization`, `test_model_selection`, `test_device`, `test_metrics`, `test_physics_checks`, `test_uncertainty`, `test_mittag_leffler`, `test_prony`, `test_data_quality`, `test_compatibility`, `test_protocol_preprocessing`）以及各模型核（`test_epm_kernels*`, `test_hl_kernels`, `test_sgr_kernels`, `test_sgr_monte_carlo`, `test_sgr_population_balance`, `test_mct_kernels`, `test_spp_kernels`, `test_structure_factor`, `test_volterra_creep`）、初始化（`test_fractional_initializers`, `test_base_initializer`, `test_auto_p0`, `test_initialization`, `test_initialization_backward_compat`）、其他专项回归（`test_gmm_alias_and_decay`, `test_ml_convergence`, `test_placeholder_model_guard`）。

## 常见问题 (FAQ)

- **为什么初始化启发式按模型族拆分文件？** 每个分数阶模型的参数空间与合理初值范围差异很大；集中在一个文件会让 `auto_p0.py` 的分发逻辑难以维护。新增分数阶模型需在 `initialization/` 添加对应文件并在 `auto_p0.py` 中注册。

## 相关文件清单

`device.py`, `data_quality.py`, `epm_kernels.py`(+`_tensorial`), `compatibility.py`, `hl_kernels.py`, `mittag_leffler.py`, `prony.py`, `metrics.py`, `mct_kernels.py`, `optimization.py`, `physics_checks.py`, `model_selection.py`, `sgr_kernels.py`, `sgr_monte_carlo.py`, `sgr_population_balance.py`, `protocol_preprocessing.py`, `uncertainty.py`, `volterra_creep.py`, `spp_kernels.py`, `structure_factor.py`, `initialization/*.py`(14 个文件)

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（架构扫描）。
