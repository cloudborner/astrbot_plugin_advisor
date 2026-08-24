# 部署、验证与回滚

## 部署前只读检查

1. 记录 `free -h`、Swap、磁盘、CPU 和 `docker stats --no-stream`。
2. 确认 AstrBot 容器名称、镜像、版本、插件数据挂载和当前插件清单。
3. 保存 `data/plugins` 目录及 AstrBot 配置备份。
4. 确保至少 384 MiB 可用内存和可用 Swap；低于阈值时不部署。

## 部署

1. 本地完成单元测试、compileall、索引质量验证和签名验证。
2. 打包时排除 `.cache`、`.ruff_cache`、`__pycache__`、测试私钥和本地统计。
3. 上传到临时目录，核对 SHA-256 后再复制到 AstrBot 插件目录。
4. 安装 `requirements.txt`，仅重载顾问插件；必要时再重启 AstrBot 容器。
5. 不在生产机安装市场全部第三方插件进行测试。

## 验收

- AstrBot 日志无 import、schema、签名或 OOM 错误。
- `/插件体检` 正确显示 cgroup/主机余量和画像数量。
- `/插件推荐` 能离线工作，返回分项分、置信度和风险。
- `/资源画像 <已知插件>` 返回版本绑定、证据和未知项。
- 构造不存在/过期画像时，GitHub 回退失败也能保守返回市场画像。
- 使用篡改 manifest/索引测试时拒绝更新且旧索引保持可用。
- 监测启动后 10 分钟 RSS、CPU、Swap 和容器重启次数；不得触发 OOM。

## 回滚

1. 禁用或移走 `astrbot_plugin_advisor` 目录。
2. 若只需回滚索引，把 `resource_profiles.json.bak` 恢复为当前索引。
3. 重载插件或重启 AstrBot 容器。
4. 验证原有插件清单、聊天平台连接和容器重启计数。

插件不修改数据库 schema、不安装推荐插件、不改 LLBot/AstrBot 业务配置；因此源码目录与数据目录备份足以恢复。部署记录和实际服务器测量结果应写入 `docs/test-report.md`。
