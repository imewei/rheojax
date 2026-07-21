[根目录](../../CLAUDE.md) > **models**

# models — 53 个流变本构模型 / 22 个模型族

## 模块职责

实现全部流变本构模型，每个模型继承 `rheojax.core.base.BaseModel`，通过 `@ModelRegistry.register` 装饰器注册到全局 `Registry`。涵盖经典线性粘弹性（Maxwell/Zener/SpringPot）、分数阶模型族（11 个）、非牛顿流动模型（6 个）、多模式（GeneralizedMaxwell/Prony）、软玻璃流变学（SGR）、STZ、Fluidity(+Saramito EVP)、DMT 触变、SPP 屈服应力、ITT-MCT、FIKH、Giesekus、TNT（瞬态网络）、VLB、IKH、HVM/HVNM（乙烯基聚合物）、Hébraud-Lequeux、EPM（弹塑性介观）模型。

## 入口与启动

- **懒加载入口**：`rheojax/models/__init__.py` 通过 `_LAZY_IMPORTS` 字典 + 模块级 `__getattr__` 实现按需导入（避免启动时导入全部 53 个模型及 scipy/numpyro 依赖，节省约 270ms）。
- **强制全量注册**：`_ensure_all_registered()`（幂等）— CLI `inventory` 命令、测试 conftest、`ModelRegistry.create(name)` 按注册名创建时需要调用此函数以确保 registry 完整。
- 典型子包入口文件：`<family>/_base.py`（抽象基类/共享逻辑）、`<family>/_kernels.py`（JAX 数值核）、`<family>/local.py` 或 `<family>.py`（0D/局部模型实现）、`<family>/nonlocal_model.py`（1D PDE 非局部变体，如 fluidity, dmt, vlb）。

## 对外接口

用法示例（见 `__init__.py` 顶部 docstring）：
```python
from rheojax.models import Maxwell, FractionalZenerSolidLiquid, PowerLaw
model = Maxwell()
from rheojax.core.registry import ModelRegistry
model = ModelRegistry.create('maxwell')
models = ModelRegistry.list_models()
```
每个模型公开：`fit()`（NLSQ）、`predict()`、继承自 `BayesianMixin` 的 `fit_bayesian()`/`sample_prior()`/`get_credible_intervals()`。多模式模型（GeneralizedMaxwell、TNT、VLB 系列）额外支持 warm-start 与元素数自动优化。

## 关键依赖与配置

- `nlsq`（两步 NLSQ 拟合 + softmax 惩罚，见 `multimode/generalized_maxwell.py`）、`diffrax`（瞬态 ODE 求解，懒加载）、`interpax`（JIT 安全插值）。
- 依赖 `rheojax.core`（BaseModel/Registry/TestMode）、`rheojax.utils`（`prony.py`、`optimization.py`、`initialization/`、各模型专属 kernel 辅助如 `epm_kernels.py`/`sgr_kernels.py`/`mct_kernels.py`）。
- 无独立配置文件；模型行为通过构造参数与 `ParameterSet` 约束控制。

## 数据模型

- 多数模型的核心状态是 `ParameterSet`（见 core 模块），部分多网络模型（VLBMultiNetwork、TNTMultiSpecies、FMLIKH）使用 `SharedParameterSet` 管理跨模式共享参数。
- 非局部（PDE）模型（`fluidity/nonlocal_model.py`、`dmt/nonlocal_model.py`、`vlb` nonlocal）额外维护空间离散网格状态。

## 测试与质量

- 测试镜像结构：`tests/models/<family>/test_*.py`，约 90 个测试文件，覆盖每个模型族（classical, fractional, flow, fluidity(+saramito), dmt, multimode, giesekus, ikh, itt_mct, tnt, vlb, sgr, stz, hvm, hvnm, hl, epm, fikh, spp）。
- 额外测试：`test_model_imports.py`（懒加载完整性）、`multimode/test_gmm_bayesian_safety.py`（GMM 贝叶斯先验安全阈值）。
- 已知修复历史：verification 套件曾发现 FM-on-solid-data 发散（→FKV swap）与 `_coerce_pred` no-op 缺陷（见项目 memory `verification_template_defects`）。

## 常见问题 (FAQ)

- **新增模型如何接入？** 在对应家族目录下实现类（继承 `BaseModel`），加 `@ModelRegistry.register(name=...)`，然后在 `rheojax/models/__init__.py` 的 `_LAZY_IMPORTS` 中添加 `"ClassName": ("rheojax.models.<family>", "ClassName")` 并追加到 `__all__`。
- **为什么 `import rheojax.models` 后 `ModelRegistry.list_models()` 是空的？** 懒加载不会触发注册装饰器；需调用 `rheojax.models._ensure_all_registered()`。

## 相关文件清单（按家族，105 个源文件）

`classical/`(3), `fractional/`(11+mixin), `flow/`(6), `multimode/`(GeneralizedMaxwell), `sgr/`(2), `stz/`(1+kernels), `fluidity/`(2)+`fluidity/saramito/`(2), `dmt/`(2), `spp/`(1), `itt_mct/`(2), `fikh/`(2+caputo/thermal), `giesekus/`(2), `tnt/`(5), `vlb/`(4), `ikh/`(2), `hvm/`(1), `hvnm/`(1), `hl/`(1), `epm/`(2)。深度扫描重点 `multimode/generalized_maxwell.py`：实现 Prony 级数、tri-mode 一致性（松弛/振荡/蠕变）、两步 NLSQ + softmax 惩罚、透明元素最小化、NumPyro NUTS 热启动。

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（架构扫描）。
