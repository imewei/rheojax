[根目录](../../CLAUDE.md) > **logging**

# logging — 结构化日志

## 模块职责

为全仓提供统一的结构化日志系统，供 `core`/`cli`/`gui`/`pipeline` 等模块记录拟合进度、诊断信息与操作审计。支持标准/详细/JSON/科学计数法多种输出格式，专属上下文管理器覆盖拟合、贝叶斯采样、变换、I/O、流水线阶段、GUI 操作等常见场景。

## 入口与启动

- `rheojax/logging/__init__.py` 汇总全部公共 API；典型用法：
  ```python
  from rheojax.logging import configure_logging, get_logger
  configure_logging(level="INFO")
  logger = get_logger(__name__)
  logger.info("Starting analysis", model="Maxwell")
  ```
- `config.py` — `configure_logging()`（全局配置入口，幂等）、`LogConfig`、`LogFormat`（枚举）、`get_config()`、`is_configured()`、`reset_config()`（主要供测试隔离用）。
- `logger.py` — `get_logger(name)`（带缓存的 logger 工厂）、`RheoJAXLogger`、`clear_logger_cache()`。

## 对外接口

- **配置**：`LogConfig`, `LogFormat`, `configure_logging`, `get_config`, `is_configured`, `reset_config`（`config.py`）。
- **Logger**：`RheoJAXLogger`, `get_logger`, `clear_logger_cache`（`logger.py`）。
- **上下文管理器**（`context.py`）：`log_operation`（通用）、`log_fit`、`log_bayesian`、`log_transform`、`log_io`、`log_pipeline_stage`、`log_gui_action`——均以 `with log_fit(logger, "Maxwell", data_shape=(100,)) as ctx: ... ctx["R2"] = result.r_squared` 形式使用，退出时自动记录耗时/异常。
- **格式化器**（`formatters.py`）：`StandardFormatter`, `DetailedFormatter`, `JSONFormatter`, `ScientificFormatter`（科学计数法数值格式化，适配拟合参数输出）、`get_formatter(name)` 工厂函数。
- **处理器**（`handlers.py`）：`RheoJAXStreamHandler`, `RheoJAXRotatingFileHandler`, `RheoJAXMemoryHandler`（内存环形缓冲，供 GUI 日志面板读取）, `NullHandler`, `create_handlers()`。

## 关键依赖与配置

- 仅依赖标准库 `logging`；无第三方依赖。
- **环境变量**：`RHEOJAX_LOG_LEVEL`（DEBUG/INFO/WARNING/ERROR）、`RHEOJAX_LOG_FILE`（指定路径启用文件日志）、`RHEOJAX_LOG_FORMAT`（standard/detailed/json）。
- `configure_logging()` 应在应用启动早期调用一次（CLI `main.py`、GUI `main.py` 均在启动时调用）；重复调用需先 `reset_config()`（主要用于测试）。

## 数据模型

- 无持久化数据模型；`LogConfig` 为不可变配置快照（level/format/file path）。
- `RheoJAXMemoryHandler` 内部维护有限长度的日志记录环形缓冲区，供 GUI `widgets/log_dock.py` 拉取展示。

## 测试与质量

- `tests/logging/`（4 个文件）：`test_config`（配置/环境变量解析）、`test_logger`（logger 缓存/获取）、`test_context`（各上下文管理器耗时/异常记录行为）、`test_logging_integration`（跨模块集成，验证 core/cli 日志实际落地）。

## 常见问题 (FAQ)

- **如何在拟合过程中记录耗时与结果？** 用 `with log_fit(logger, model_name, data_shape=...) as ctx:` 包裹拟合调用，退出时自动记录耗时；可在 `ctx` 字典中追加自定义字段（如 `ctx["R2"] = ...`），一并写入日志记录。
- **GUI 日志面板如何拿到日志？** `gui/widgets/log_dock.py` 通过 `RheoJAXMemoryHandler` 的环形缓冲读取最近日志，无需轮询文件。
- **多次调用 `configure_logging()` 会怎样？** 默认幂等（`is_configured()` 为真时直接跳过），测试中需要重新配置时应先调用 `reset_config()`。

## 相关文件清单

`config.py`, `logger.py`, `context.py`, `formatters.py`, `handlers.py`, `__init__.py`

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（补扫，此前根架构总览中标注为"未生成独立文档"）。
