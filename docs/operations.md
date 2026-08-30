# 部署、验证与回滚

## 全市场源码功能索引

在 Windows 上双击 `scripts/build_full_plugin_index.cmd`，或在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\build_full_plugin_index.py
```

流水线依次读取市场快照、计算公开 GitHub 仓库、断点下载、安全解压、删除已成功解压的压缩包、静态提取功能与资源证据、合并市场资料、生成能力索引并自动校验。插件源码只作为不可信文本和 AST 读取，不会被安装、导入或执行。

断点数据保存在 `source_extracted/pipeline_manifest.json`。中断后重新运行同一命令即可；已经成功解压且版本未变化的项目不会再次下载。固定提交返回 404 时会尝试公开默认分支，并在清单中标记 `used_default_branch_fallback`，不会把默认分支伪装成原提交。

常用参数：

- `--plan`：只生成下载计划，不联网。
- `--workers 4`：下载和解压并发数，范围 1–16。
- `--proxy-url http://127.0.0.1:7897`：显式使用代理；默认忽略系统代理。
- `--keep-archives`：成功解压后仍保留压缩包；默认删除。
- `--max-archive-mib`、`--max-plugin-mib`、`--max-total-gib`、`--minimum-free-gib`：磁盘和解压安全限制。

主要输出为 `data/source_function_evidence.json`、`data/source_resource_profiles.json`、`data/source_resource_index.json` 和 `data/plugin_capabilities.json`；最终汇总写入 `artifacts/full_source_pipeline_report.json`。原始源码、压缩包和功能证据明细不会打入发布包。

## 部署前只读检查

1. 记录 `free -h`、Swap、磁盘、CPU 和 `docker stats --no-stream`。
2. 确认 AstrBot 容器名称、镜像、版本、插件数据挂载和当前插件清单。
3. 保存 `data/plugins` 目录及 AstrBot 配置备份。
4. 确保至少 384 MiB 可用内存和可用 Swap；低于阈值时不部署。

## 部署

1. 本地完成单元测试、Ruff、compileall、索引质量验证和发布包测试；维护签名索引的离线环境需另外安装 `cryptography`。
2. 使用 `scripts/build_release.py` 打包；脚本会排除缓存、测试、Schema、签名维护模块、源码扫描中间产物和开发配置。
3. 上传到临时目录，核对 SHA-256 后再复制到 AstrBot 插件目录。
4. 安装 `requirements.txt`，仅重载顾问插件；必要时再重启 AstrBot 容器。
5. 不在生产机安装市场全部第三方插件进行测试。

## 验收

- AstrBot 日志无 import、schema 或 OOM 错误。
- `/插件体检` 正确显示 cgroup/主机余量和画像数量。
- `/需求分析` 能完成读取、词组确认、模型分析和候选复核；群号只显示为报告小字标注。
- `/最近需求分析` 能看到最近报告摘要，`/重发需求分析` 不触发模型调用且能重新发送同一报告；检查点文件不含明文群号和原始聊天。
- 支持 JSON Schema 的 Provider 收到 `response_format`；明确不支持的 Provider 只兼容降级一次，随后仍执行本地严格解析和证据校验。
- `/资源画像 <已知插件>` 返回版本绑定、证据和未知项。
- 构造不存在/过期画像时，GitHub 回退失败也能保守返回市场画像。
- 监测启动后 10 分钟 RSS、CPU、Swap 和容器重启次数；不得触发 OOM。

## 回滚

1. 禁用或移走 `astrbot_plugin_advisor` 目录。
2. 若只需回滚索引，把 `resource_profiles.json.bak` 恢复为当前索引。
3. 重载插件或重启 AstrBot 容器。
4. 验证原有插件清单、聊天平台连接和容器重启计数。

插件不修改数据库 schema、不安装推荐插件、不改 LLBot/AstrBot 业务配置；因此源码目录与数据目录备份足以恢复。部署记录和实际服务器测量结果应写入 `docs/test-report.md`。
