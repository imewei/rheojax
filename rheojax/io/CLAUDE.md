[根目录](../../CLAUDE.md) > **io**

# io — 统一文件 I/O

## 模块职责

提供流变数据的读写：TA Instruments TRIOS（Excel/CSV/TXT/JSON）、Anton Paar、通用 CSV/Excel、多文件加载器（TTS/SRFS/系列）、HDF5/Excel/NPZ 写入器、格式自动检测、SPP 分析结果的 MATLAB 兼容导出（CSV/HDF5/TXT）。

## 入口与启动

- `rheojax/io/__init__.py` 汇总公共 API；无 CLI 独立入口（供 `rheojax.cli.cmd_load`/`cmd_export` 调用）。
- 自动格式检测：`readers/auto.py` 的 `auto_load()`。
- TRIOS 专属子包：`readers/trios/`（`csv.py`, `excel.py`, `json.py`, `txt.py`, `common.py`, `schema/`）。

## 对外接口

- **读取器**：`load_trios`, `load_csv`, `load_excel`, `load_anton_paar`, `auto_load`（`rheojax.io.readers`）。
- **多文件加载**：`load_tts`, `load_srfs`, `load_series`（`readers/multi_file.py`）。
- **写入器**：`save_hdf5`, `load_hdf5`, `save_excel`, `save_npz`, `load_npz`（`rheojax.io.writers`）。
- **SPP 导出**：`export_spp_txt`, `export_spp_hdf5`, `export_spp_csv`, `to_matlab_dict`（`spp_export.py`）。
- **分析导出**：`AnalysisExporter`（`analysis_exporter.py`）、`NumpyJSONEncoder`（`json_encoder.py`，处理 NumPy 类型的 JSON 序列化）。
- **异常/校验**：`RheoJaxFormatError`, `RheoJaxValidationWarning`（`_exceptions.py`）；`validate_protocol()`（`readers/_validation.py`）。
- **列映射**：`readers/_column_mapping.py`（供 GUI `column_mapper` 对话框复用）。

## 关键依赖与配置

- `h5py`（HDF5）、`openpyxl`/`xlrd`（Excel 读写）、`pandas`（表格解析）。
- 依赖 `rheojax.core.data.RheoData` 作为统一返回容器；依赖 `rheojax.core.test_modes` 做协议校验。

## 数据模型

- 所有读取函数最终返回 `RheoData`（或其列表，多文件场景），附带测试模式/单位元数据。
- TRIOS schema 子模块（`readers/trios/schema/experiment.py`, `dataset.py`）定义 TRIOS 导出文件的结构化模式。

## 测试与质量

- `tests/io/`（23 个文件）：格式专项（`test_anton_paar`, `test_csv_detection`, `test_trios_real_data`, `test_trios_chunked`/`_auto_chunk`/`_chunked_integrity`, `test_trios_memory_profiling`）、写入器（`test_writers`, `test_npz_writer`）、`test_multi_file`, `test_unit_conversion`, `test_protocol_metadata`, `test_column_mapping`, `test_validation`, `test_spp_export`, `test_analysis_exporter`, `test_dmta_rejection`, `test_io_fixes`。
- TRIOS 子测试：`tests/io/readers/trios/test_csv.py`, `test_json.py`。

## 常见问题 (FAQ)

- **大文件如何处理？** TRIOS 读取器支持分块读取（chunked），见 `test_trios_chunked*` 测试；避免一次性载入超大数据集导致内存问题。
- **格式识别失败怎么办？** 使用 `auto_load()` 触发自动检测；失败时抛出 `RheoJaxFormatError`，检查 `readers/_validation.py` 的协议校验逻辑。

## 相关文件清单

`analysis_exporter.py`, `json_encoder.py`, `_exceptions.py`, `spp_export.py`, `readers/{csv_reader,excel_reader,anton_paar,auto,multi_file,_column_mapping,_utils,_validation}.py`, `readers/trios/{csv,excel,json,txt,common}.py`, `readers/trios/schema/{experiment,dataset}.py`, `writers/{hdf5_writer,excel_writer,npz_writer}.py`

## 变更记录 (Changelog)

- 2026-07-17: 初始生成（架构扫描）。
