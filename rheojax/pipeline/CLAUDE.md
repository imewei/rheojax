[根目录](../../CLAUDE.md) > **pipeline**

# pipeline — 流式工作流 API

## 模块职责

提供链式调用（fluent API）风格的流变分析工作流封装：数据加载 → 拟合（NLSQ/贝叶斯）→ 可视化 → 保存。包含预置工作流（主曲线构建、模型比较、蠕变↔松弛转换、频域↔时域转换、SPP 幅度扫描）、可编程构建器、批处理入口。

## 入口与启动

- `rheojax/pipeline/base.py` — `Pipeline` 核心类：`.load()`, `.fit()`, `.plot()`, `.save()` 链式方法。
- `rheojax/pipeline/bayesian.py` — `BayesianPipeline`：`.fit_nlsq()` → `.fit_bayesian()` → `.plot_posterior()` → `.save()`。
- `rheojax/pipeline/workflows.py` — 预置工作流类：`MastercurvePipeline`, `ModelComparisonPipeline`, `CreepToRelaxationPipeline`, `FrequencyToTimePipeline`, `SPPAmplitudeSweepPipeline`。
- `rheojax/pipeline/builder.py` — `PipelineBuilder`：`.add_load_step()`, `.add_transform_step()`, `.add_fit_step()`, `.build()`（编程式流水线构建，供 CLI `run pipeline.yaml` 与 GUI pipeline 模式复用）。
- `rheojax/pipeline/batch.py` — `BatchPipeline`：`.process_directory()`, `.export_summary()`（批量处理多个数据文件）。

## 对外接口

见 `__init__.py` 顶部 docstring 中的四个用例：basic workflow、Bayesian workflow、model comparison、batch processing、pipeline builder。所有工作流最终产出 `FitResult`/`ModelComparison`（core 模块）或导出文件（HDF5/Excel）。

## 关键依赖与配置

- 依赖 `rheojax.core`（`BaseModel`, `FitResult`, `ModelComparison`）、`rheojax.io`（加载/保存）、`rheojax.models`/`rheojax.transforms`（注册表驱动的模型/变换实例化）。
- `PipelineBuilder._validate_components()` 会同时调用 `rheojax.models._ensure_all_registered()` 与 `rheojax.transforms._ensure_all_registered()` 校验流水线步骤引用的组件存在。
- 与 `rheojax.parallel` 集成：批处理/主曲线/模型比较可并行执行（见测试 `test_batch_parallel.py`, `test_mastercurve_parallel.py`, `test_model_comparison_parallel.py`）。

## 数据模型

- Pipeline 内部维护有序的 step 列表（load/transform/fit/plot/save），每步操作 `RheoData` 或其派生结果；`BatchPipeline` 维护每文件一份结果字典供 `export_summary()` 汇总为 Excel。

## 测试与质量

- `tests/pipeline/`（16 个文件）：`test_pipeline_base`, `test_bayesian_pipeline`, `test_builder`(+`_extensions`), `test_batch`(+`_parallel`, `_vectorize`), `test_workflows`, `test_mastercurve_parallel`, `test_model_comparison_parallel`, `test_arviz_diagnostics`, `test_pipeline_visualization`, `test_spp_pipeline_defaults`, `test_removed_dmta_kwargs`, `test_device_memory`。

## 常见问题 (FAQ)

- **如何新增预置工作流？** 在 `workflows.py` 中新增类，遵循现有工作流的构造签名，并在 `__init__.py` 的 `__all__` 中导出。
- **批处理与并行池的关系？** `BatchPipeline`/工作流并行变体委托给 `rheojax.parallel.parallel_map`/`parallel_load`；顺序模式可用 `RHEOJAX_SEQUENTIAL=1` 强制关闭并行以便调试。

## 相关文件清单

`base.py`, `bayesian.py`, `workflows.py`, `builder.py`, `batch.py`, `__init__.py`

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（架构扫描）。
