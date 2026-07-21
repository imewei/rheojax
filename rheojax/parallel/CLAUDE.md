[根目录](../../CLAUDE.md) > **parallel**

# parallel — 进程级并行执行层

## 模块职责

为模型拟合、贝叶斯推断和数据变换提供基于进程的并行化；I/O 场景使用线程并行。通过环境变量控制并行策略（worker 数量、隔离方式、是否禁用并行、是否预热进程池）。

## 入口与启动

- `rheojax/parallel/__init__.py` 汇总公共 API；`PersistentProcessPool` 通过 `__getattr__` 懒加载（避免包级别引入 `multiprocessing`）。
- `pool.py` — `PersistentProcessPool`：常驻进程池实现，避免每次拟合重新 fork/spawn 进程的开销。
- `api.py` — `parallel_load()`（多线程并行加载文件）、`parallel_map()`（多进程并行映射，用于批量拟合/变换）。
- `config.py` — `configure()`, `get_default_workers()`, `get_parallel_config()`, `get_worker_isolation()`, `is_sequential_mode()`。

## 对外接口

```python
from rheojax.parallel import configure, parallel_load, parallel_map
datasets = parallel_load(['data1.csv', 'data2.csv'], x_col='time', y_col='stress')
results = list(parallel_map(fit_func, items, n_workers=4))
configure(n_workers=4, warm_pool=True)
```

## 关键依赖与配置

环境变量（`config.py` 读取）：
- `RHEOJAX_PARALLEL_WORKERS=N` — worker 数量（默认自动检测 CPU 核数）。
- `RHEOJAX_SEQUENTIAL=1` — 禁用全部并行（调试/CI 确定性场景）。
- `RHEOJAX_WORKER_ISOLATION=subprocess|thread` — 隔离级别。
- `RHEOJAX_WARM_POOL=1` — 启动时预初始化 worker（降低首次拟合延迟）。
- 标准库 `multiprocessing`（懒加载以避免包导入开销）。

## 数据模型

- 无持久化数据模型；`PersistentProcessPool` 内部维护 worker 进程句柄与任务队列状态。

## 测试与质量

- `tests/parallel/`（8 个文件）：`test_api`, `test_pool`, `test_config`, `test_warmup`, `test_integration`, `test_imports`，`conftest.py` 提供并行相关 fixture。
- 与 `pipeline`/`models` 的并行集成分别在 `tests/pipeline/test_batch_parallel.py`, `test_mastercurve_parallel.py`, `test_model_comparison_parallel.py` 中验证。

## 常见问题 (FAQ)

- **GUI/notebook 环境中并行卡死？** 优先检查是否需要 `RHEOJAX_SEQUENTIAL=1` 或 `RHEOJAX_WORKER_ISOLATION=thread`（子进程在某些交互环境下可能与事件循环冲突）。
- **为什么用进程而不是线程做计算并行？** JAX/NLSQ 拟合是 CPU/GPU 密集型，Python GIL 下线程无法真正并行；I/O（文件加载）则用线程避免进程间序列化开销。

## 相关文件清单

`pool.py`, `config.py`, `api.py`, `__init__.py`

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（架构扫描）。
