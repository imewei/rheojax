[根目录](../../CLAUDE.md) > **cli**

# cli — 命令行工具

## 模块职责

提供 `rheojax`/`rj` 命令行入口，封装 NLSQ 拟合、贝叶斯推断、SPP 分析、数据加载/变换/导出、YAML 流水线执行与批处理。

## 入口与启动

- **console_scripts**：`rheojax = rheojax.cli.main:cli_entry`、`rj = rheojax.cli.main:cli_entry`（见 `pyproject.toml`）。
- `main.py`：`create_main_parser()` 定义子命令 argparse 解析器；`main(args)` 按 `args[0]` 分发到子模块；`cli_entry()` 包装 `sys.exit(main())`。
- 子命令模块（各自有独立 `main(args)`）：`fit.py`, `bayesian.py`, `spp.py`, `cmd_load.py`, `cmd_transform.py`, `cmd_export.py`, `cmd_run.py`, `cmd_pipeline.py`, `cmd_batch.py`。
- `info`/`inventory` 子命令直接在 `main.py` 内实现（`show_info()`, `show_inventory()`）。

## 对外接口（子命令）

```
rheojax fit data.csv --model maxwell --x-col time --y-col G_t
rheojax bayesian data.csv --model maxwell --x-col time --y-col G_t --warm-start
rheojax spp analyze data.csv --omega 1.0 --gamma-0 0.5
rheojax load data.csv --json | rheojax transform fft_analysis --input -
rheojax export analysis.h5 --output bundle.xlsx --format excel
rheojax run pipeline.yaml
rheojax pipeline init --template basic --output pipeline.yaml
rheojax batch "data/*.csv" --model maxwell --test-mode relaxation
rheojax info
rheojax inventory [--protocol laos] [--type spectral]
```
- `inventory` 命令会强制调用 `rheojax.models._ensure_all_registered()` 与 `rheojax.transforms._ensure_all_registered()`，因为懒加载不会自动填充 registry。

## 关键依赖与配置

- `argparse`（标准库）、`rich`（格式化输出，`_output.py`）、`pyyaml`（`_yaml_runner.py`/`_yaml_schema.py` 解析 YAML 流水线配置）。
- 依赖 `rheojax.logging.configure_logging()`/`get_logger()` 做结构化日志；依赖 `rheojax.core.registry.Registry`、`rheojax.core.jax_config.safe_import_jax`。
- `_envelope.py`：定义 CLI 间传递数据的 JSON "信封" 格式（`load --json | transform --input -` 管道模式）。
- `_globals.py`：跨子命令共享的全局选项。
- `_templates.py`：`pipeline init --template` 的模板定义。

## 数据模型

- CLI 间数据交换通过 JSON envelope（`_envelope.py`）传递 `RheoData` 的序列化形式，支持 Unix 管道风格组合（`load | transform`）。
- YAML 流水线配置遵循 `_yaml_schema.py` 定义的 schema，由 `_yaml_runner.py` 执行。

## 测试与质量

- `tests/cli/`（17 个文件）：`test_main_dispatch`（分发逻辑）、各子命令（`test_cmd_load`, `test_cmd_transform`, `test_cmd_export`, `test_cmd_run`, `test_cmd_pipeline`, `test_cmd_batch`, `test_bayesian`, `test_spp`）、`test_envelope`, `test_yaml_schema`, `test_yaml_runner`, `test_globals`, `test_templates`, `test_output`, `test_dmta_cli`。
- mypy：`rheojax.cli.*` 在 `pyproject.toml` 中被标记 `ignore_errors = true`（PySide6/CLI 桩问题跨模块共享豁免配置）。

## 常见问题 (FAQ)

- **子命令找不到模型/变换？** 确认已触发 `_ensure_all_registered()`；直接调用 `ModelRegistry`/`TransformRegistry` 而不先导入并注册子模块会返回空列表。
- **YAML 流水线校验失败？** 检查 `_yaml_schema.py` 的 schema 定义与 `cmd_pipeline.py` 的 `validate` 子命令。

## 相关文件清单

`main.py`, `fit.py`, `bayesian.py`, `spp.py`, `cmd_load.py`, `cmd_transform.py`, `cmd_export.py`, `cmd_run.py`, `cmd_pipeline.py`, `cmd_batch.py`, `_yaml_runner.py`, `_yaml_schema.py`, `_globals.py`, `_output.py`, `_envelope.py`, `_templates.py`, `__init__.py`

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（架构扫描）。
