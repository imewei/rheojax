[根目录](../../CLAUDE.md) > **gui**

# gui — PySide6 桌面应用

## 模块职责

Qt（PySide6）图形界面，提供交互式数据加载/可视化、模型拟合（实时参数更新）、贝叶斯推断（ArviZ 诊断）、变换流水线（主曲线/FFT/SRFS）、多视图绘图与出版级导出、GPU 加速状态监控。架构为 Redux 风格状态管理（Qt 信号驱动）+ 服务层（对接 rheojax 核心 API）+ 后台 Worker（长时间计算不阻塞 UI）。

## 入口与启动

- **console_scripts**：`rheojax-gui = rheojax.gui.main:main`、`rj-gui = rheojax.gui.main:main`。
- `rheojax/gui/__init__.py` — 包级 `main()` 便捷包装，转发到 `gui/main.py`。
- `gui/main.py` — 真正的启动逻辑：解析 `--project`/`--import`/`--verbose`，依赖检查，日志配置，创建 QApplication 与主窗口，运行事件循环。
- **单一外壳架构**：`gui/workspace/window.py` — `WorkspaceWindow`（三步向导模式：`MODES = ("fit", "transform", "pipeline")`）是**唯一**的 GUI 入口。CHANGELOG（`[Unreleased]`）记录了一次 **BREAKING** 变更：遗留 `RheoJAXMainWindow` 外壳及 `--legacy`/`--workspace` CLI 标志已被移除；`WorkspaceWindow` 早已是默认外壳，现在是唯一外壳。遗留外壳暴露的能力（`--project`, `--import`/`--protocol`, `--maximized`）均已在当前唯一路径上完整支持。（本仓库项目 memory 中关于"双外壳/`--legacy`"的记录已过期，本次扫描通过 grep 全仓 + 读取 CHANGELOG.md 确认移除，已在此更正。）

## 对外接口

- **状态管理**（`gui/state/`）：`store.py`（中心 store）、`actions.py`、`selectors.py`、`signals.py`、`reducers/`（按域拆分：`fitting_reducers`, `project_reducers`, `model_reducers`, `ui_reducers`, `pipeline_reducers`, `data_reducers`, `bayesian_reducers`）。
- **服务层**（`gui/services/`）：`data_service.py`, `export_service.py`, `bayesian_service.py`, `pipeline_execution_service.py`, `transform_service.py`, `plot_service.py`, `model_service.py` — 封装对 `rheojax.core`/`rheojax.pipeline`/`rheojax.io` 的调用，是 GUI 与核心库之间的唯一桥梁。
- **后台 Worker**（`gui/jobs/`）：`fit_worker.py`, `bayesian_worker.py`, `import_worker.py`, `export_worker.py`, `transform_worker.py`, `preview_worker.py`, `worker_pool.py`, `cancellation.py`, `subprocess_fit.py`/`subprocess_bayesian.py`（子进程隔离执行）, `process_adapter.py`, `_cleanup.py`。
- **Foundation**（`gui/foundation/`）：`state.py`（`AppState`/`FitState`/`NlsqConfig`/`NutsConfig` 等 dataclass）、`contract.py`（协议必需字段声明）、`library.py`（`DatasetLibrary`）、`notifier.py`（`DatasetLibraryNotifier`）、`pipeline_bridge.py`、`project_codec.py`（项目保存/加载序列化）、`priors.py`、`import_service.py`、`metrics.py`、`invalidation.py`（缓存失效级联）。
- **Workspace 向导步骤**：
  - `workspace/fit/`：`step1_protocol_model.py` → `step2_data.py` → `step3_nlsq.py` → `step4_nuts.py` → `step5_visualize.py` → `step6_export.py`，由 `fit_controller.py`（`build_fit_controller()`）编排。
  - `workspace/transform/`：`step1_pick.py` → `step2_slots.py`（`slots_spec.py`）→ `step3_run.py` → `step4_visualize.py` → `step5_export.py`，由 `transform_controller.py` 编排。
  - `workspace/pipeline/`：`step1_configure_run.py`、`controller.py`、`batch_runner.py`、`cancel_runnable.py`、`models.py`。
  - 共用组件：`library_rail.py`（数据集库侧栏）、`inspector.py`、`status_bar.py`、`stepper_canvas.py`。
- **对话框**（`gui/dialogs/`）：`import_wizard.py`, `column_mapper.py`, `dataset_preview.py`, `fitting_options.py`, `bayesian_options.py`, `export_options.py`, `pipeline_templates.py`, `preferences.py`, `about.py`。
- **控件**（`gui/widgets/`）：`plot_canvas.py`, `pyqtgraph_canvas.py`（GPU 加速绘图）, `arviz_canvas.py`, `base_arviz_widget.py`, `parameter_form.py`, `parameter_table.py`, `priors_editor.py`, `dropdown.py`, `log_dock.py`。
- **兼容层**：`gui/compat.py`（PyQt/PySide6 抽象，配合 `qtpy`）。

## 关键依赖与配置

- `PySide6>=6.10.2`、`qtpy`（绑定抽象）、`pyqtgraph`（交互式 GPU 绘图）、`qasync`（异步事件循环整合）。
- `gui/utils/config.py`（用户偏好持久化）、`gui/resources/styles/tokens.py`（`ThemeManager`，明暗主题）、`gui/utils/_dependency_guard.py`（可选依赖缺失时的优雅降级）。
- mypy：`rheojax.gui.*` 在 `pyproject.toml` 中 `ignore_errors = true`（PySide6 类型桩不完整）。

## 数据模型

- `AppState`（`foundation/state.py`）是顶层应用状态；内含 `FitState`（`protocol`, `model_key`, `model_config`, `data_ref`, `column_map`, `control_vars`, `nlsq_config: NlsqConfig`, `nuts_config: NutsConfig`, `nlsq_result`, `nuts_result`）。
- **已知未接线字段**：`FitState.control_vars` 已在失效级联中传导（`contract.py` 声明每协议必需名如 `sigma0`/`gamma0`），但没有 GUI 步骤写入它，也没有拟合/采样调用读取它 —— 真实运行的协议 kwargs 来自 `data.metadata`（`ModelService.fit()` 的 `_PROTOCOL_KWARGS`）。构建可工作的 `control_vars` UI 前，需先协调 `contract.py` 命名与 `ModelService` 实际 kwarg 词汇（如 `"gamma_dot0"` vs `"gdot"` 不是 1:1 对应）。此为代码内 `ponytail:` 注释标注的已知缺口，本次扫描直接读取 `state.py` 源码确认，非仅凭 memory。
- `DatasetLibrary`（`foundation/library.py`）管理已加载数据集集合，通过 `DatasetLibraryNotifier` 广播变更信号。

## 测试与质量

- `tests/gui/`（119+ 个文件，含子目录 `foundation/`, `services/`, `workspace/`(+`fit/`, `pipeline/`, `transform/`), `widgets/`, `visual/`）。
- 覆盖：贝叶斯全流程/长程/视觉一致性（`test_bayesian_full_parity*`, `test_bayesian_long_parity`, `test_bayesian_visual_parity`, `test_bayesian_headless`）、崩溃防护（`test_crash_prevention`）、视觉回归（`test_visual_regression*`, `visual/test_plot_service_golden.py`，golden images，`visual` marker）、GUI 冒烟/集成（`test_gui_smoke`, `test_gui_integration`, `test_gui_performance`）、workspace 各步骤单测（`workspace/fit/test_step*.py`）、pipeline 执行服务、窗口生命周期（`test_window_*`）。
- `gui` pytest marker 允许在 PySide6 未安装环境下跳过。
- 已知历史：FreeType 字形渲染崩溃需用运行时 try/except 跳过模式（非 blanket xfail/skip，见项目 memory `feedback_freetype_skip_pattern`）；`WorkspaceWindow` 曾有 QStackedWidget 最小尺寸导致窗口管理器最大化异常的 bug（PR #62，已修复）。

## 常见问题 (FAQ)

- **启动 GUI 会得到哪个窗口？** 只有 `WorkspaceWindow`（三步向导：fit/transform/pipeline）。遗留 `RheoJAXMainWindow` 及 `--legacy`/`--workspace` 标志已在 `[Unreleased]` 版本中作为 BREAKING CHANGE 移除；传入这些标志会得到 argparse "unrecognized arguments" 错误。
- **为什么拟合逻辑不直接写在 Widget 里？** 架构要求 View（PySide6 widget/dialog）与业务逻辑解耦；数值/拟合逻辑经 `gui/services/*Service` 调用 `rheojax.core`/`rheojax.pipeline`，长时间任务再委托给 `gui/jobs/*Worker` 在后台线程/子进程执行，避免阻塞 Qt 事件循环。

## 相关文件清单（115 个源文件，按子包）

`dialogs/`(10), `foundation/`(11), `jobs/`(13), `resources/`(+`styles/`)(5), `services/`(7), `state/`(+`reducers/`)(13), `utils/`(9), `widgets/`(9), `workspace/`(+`fit/`, `pipeline/`, `transform/`)(28), `compat.py`, `main.py`, `__init__.py`

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（架构扫描）。已读取 `workspace/window.py`、`foundation/state.py` 确认 `control_vars` 未接线状态；grep 全仓 + 读取 `CHANGELOG.md` 确认遗留 `RheoJAXMainWindow`/`--legacy` 外壳已在 `[Unreleased]` 移除，更正了项目 memory 中过期的"双外壳"描述。
