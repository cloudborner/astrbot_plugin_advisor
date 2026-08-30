# 本地插件源码资源扫描报告

扫描时间：2026-08-30 09:04:06 UTC
扫描模式：本地、离线、只读静态分析；未安装依赖，未导入或执行插件代码。

## 结果摘要

- 读取市场 1834 条并计算 1834 个不同 GitHub 仓库；1810 个成功断点下载、安全解压并静态扫描。
- 24 个市场链接在固定提交和公开默认分支上均返回 404；运行时索引对其保留原有元数据画像。
- 源码画像：`data/source_resource_profiles.json`（1810 条）。
- 运行时画像：`data/source_resource_index.json`（1834 条，其中源码静态画像 1810 条、元数据回退 24 条）。
- 功能证据：`data/source_function_evidence.json`（1810 条）；最终能力索引仍覆盖 1834 条。
- 人工复核队列：`data/source_resource_review_queue.json`（30 条）。
- 逐插件 LLM 资源补判默认关闭；只有管理员显式开启 `enable_llm_fallback` 时才允许使用。

## 综合等级分布

`overall_level = max(peak_memory, peak_cpu)`。

| 综合等级 | 数量 | 占比 |
| --- | ---: | ---: |
| L0 | 911 | 50.33% |
| L1 | 149 | 8.23% |
| L2 | 510 | 28.18% |
| L3 | 230 | 12.71% |
| L4 | 10 | 0.55% |

六个独立维度为 `idle_memory`、`peak_memory`、`idle_cpu`、`peak_cpu`、`disk`、`network`，每项范围 L0–L4。

## 主要规则

- 模型文件名、`torch`/`transformers` 依赖名以及普通 `generate` 字样不再证明本地模型；必须存在可达的权重加载、推理会话或模型服务启动。
- ONNX、编码器和分类器等较小本地模型通常按 L3；生成式 LLM、视觉语言模型、Whisper/CosyVoice 或本地模型服务按 L4。
- 浏览器依赖、Chrome User-Agent 和日志文字不计浏览器进程；只有实际 `launch`/WebDriver 调用计入。
- FFmpeg 只有实际子进程或 yt-dlp 后处理配置命中；FFprobe、下载器和普通子进程不会自动按多核转码计分。
- 下载主要提高磁盘和网络；只有完整读取媒体/归档/响应时提高峰值内存。
- 缓存必须位于模块、类或 AstrBot 插件主实例，且存在运行时增长写入，才可能判无界；TTL、容量、清理和淘汰逻辑视为边界。
- `while True` 仅在后台/异步语境且缺少等待、退出或并发边界时判无界；普通解析循环不再误判。
- `asyncio.to_thread`/线程池只有在压缩、解析、渲染、索引或图重建等任务上下文中提高峰值 CPU。

## 信号统计

| 信号 | 数量 |
| --- | ---: |
| 本地模型 | 22 |
| 实际浏览器启动 | 60 |
| FFmpeg | 30 |
| 下载器/下载路径 | 394 |
| 向量存储 | 20 |
| 后台任务 | 966 |
| 有界缓存 | 127 |
| 无界缓存 | 119 |
| 有界后台并发 | 926 |
| 无界后台循环/生产者 | 40 |

L4 共 10 个：`HUSTcyf/paper_rag`、`TKGEKKOU/astrbot_plugin_voice_clone_flow`、`YayiMiko/astrbot_plugin_anima_master`、`mjy1113451/astrbot_plugin_bilibili_learning_bot`、`xiaowan/astrbot_plugin_bilibili_media_parser`、`xiaowan/astrbot_plugin_bilibili_parser`、`xiewoc/astrbot_plugin_image_understanding_Janus_Pro`、`xiewoc/astrbot_plugin_image_video_understanding_Qwen2.5_VL`、`xiewoc/astrbot_plugin_spark_tts`、`xiewoc/astrbot_plugin_tts_Cosyvoice2`。这些项目均有真实生成式模型、Whisper、视觉语言模型或本地模型服务证据。

## 与旧画像的差异

在 1810 个源码绑定插件中：上调 718、不变 859、下调 233。旧报告中由依赖名、模型文件名、Chrome/FFmpeg 文本和普通缓存造成的大量 L4 已被消除。

## 验证

- 全市场流水线完成自动校验：能力与资源索引均覆盖 1834/1834 个市场版本，功能索引哈希有效。
- 功能证据覆盖 README 1803 项、命令 1450 项、配置说明 1501 项、资源静态证据 1810 项。
- 运行时索引语义校验：1834/1834 个市场版本绑定，画像哈希有效。
- 画像公式、L0–L4 范围、人工队列标记和本地模型关键证据均通过一致性检查。
- 项目测试全集运行 279 项，1 项按环境设计性跳过，另有 30 个子测试通过；Ruff 通过。
- `pytest.ini` 把测试发现限制在本仓库 `tests/`，不会收集或导入源码语料中的第三方测试。

## 解释边界

结果是容量风险静态估计，不是运行时 benchmark。动态导入、反射、配置选择、输入规模和部署环境仍可能改变真实占用；低置信度、L3/L4 与关键外部进程项目继续保留在最多 30 条的复核队列中。
