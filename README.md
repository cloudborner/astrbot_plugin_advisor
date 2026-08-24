# AstrBot 插件顾问

`astrbot_plugin_advisor` 根据服务器余量、当前群聊的去身份化聚合需求、AstrBot
官方插件市场数据和预生成的静态资源画像，对市场插件给出可解释的推荐排序。

它解决的是“这个插件是否值得在这台机器、这个群里尝试安装”，不是运行时性能测试器。
插件不会自动安装推荐结果，也不会为了评估而克隆、下载或执行第三方插件源码。

## 能力

- 对官方市场全部插件维护 `plugin_id + version + commit_sha` 绑定的资源画像。
- 分别估计空闲/峰值内存、空闲/峰值 CPU、磁盘、网络、外部进程和后台任务风险。
- 固定 100 分规则：群聊需求 30、市场使用 20、兼容性 20、资源适配 15、维护活跃 10、部署复杂度 5。
- 市场使用分严格拆成累计下载 13 分、GitHub Stars 7 分，均采用 `log1p` 后的市场百分位。
- 用去重后的有界词频、共现关系、趋势、意图和版本化主题分类识别 RoboMaster、洛克王国、人格陪伴、下载、Wiki、群管等群聊需求。
- 可按分类查看全部市场插件，或分页浏览完整市场排行，不把“前几名”伪装成完整清单。
- 检测已安装插件与候选插件在高内存峰值、外部重进程、低核后台任务方面的叠加风险。
- 画像缺失/版本落后时，只读请求 GitHub Tree/SBOM；仍不确定时可让现有大模型按严格 JSON 补充分类。
- 内置离线市场快照和资源索引；数据来源、签名校验、网络回退与安全限额由插件内部管理。
- 群聊统计默认开启且可一键关闭；只保存达到隐私阈值的按日聚合特征和本地加盐后的群标识，不保存消息原文、QQ 号或明文群号，也不读取平台昵称字段。聊天正文中的称呼可能成为聚合词项，因此这些词项仍应按潜在敏感数据保护。
- 可选模型只读取不超过 20 KiB 的去身份化结构化摘要；它可提出带特征证据的新需求，但不能指定插件、改分或安装。
- 聚合特征只在用户运行相关推荐或分析命令时发送给所选/当前模型，不会随每条消息被动外发；不希望数据离开服务器时可直接关闭群聊需求分析。

## 命令

| 命令 | 作用 |
| --- | --- |
| `/插件体检` | 显示当前容器可用内存、Swap、CPU、磁盘和索引状态。 |
| `/插件推荐 [关键词]` | 按总分从高到低推荐；不带关键词时评估全部市场插件。 |
| `/插件风险 <名称或 plugin_id>` | 显示一个插件的静态资源画像、证据、未知项和置信度。 |
| `/资源画像 <名称或 plugin_id>` | `/插件风险` 的同义命令。 |
| `/插件对比 <插件A> <插件B>` | 在当前服务器和群需求下比较两个插件。 |
| `/群需求分析` | 显示当前群去身份化聚合统计；需先启用统计。 |
| `/插件分类 [分类]` | 列出分类总览，或查看指定分类中的插件。 |
| `/插件排行 [页码]` | 分页列出官方市场的全部插件及分类、下载量和 Stars。 |
| `/刷新插件数据` | 管理员立即刷新官方市场缓存。 |

所有输出都是安装建议，不会执行安装、升级、卸载或第三方代码。

## 配置

AstrBot WebUI 只显示 4 个常用项：群聊需求分析开关、推荐数量、报告详细程度和可选模型。
其余 7 个低频选项保留在配置文件的隐藏“高级设置”中，普通配置页面不显示。无需填写市场地址、GitHub Token、
签名索引或关键词/正则规则，安装后保存默认值即可使用。完整说明见
[docs/configuration.md](docs/configuration.md)。

GitHub 增量回退不需要把 PAT 写入配置；无 Token 时可能受匿名 API 限额影响，失败会保守退化为市场元数据画像。

## 安装

从 AstrBot 插件市场安装发布版本，或把本仓库目录放入 AstrBot 的
`data/plugins/astrbot_plugin_advisor`，安装 `requirements.txt` 后重载插件。

低内存服务器建议保留至少 384 MiB 安全余量，并保持 Swap 可用。当前 1834 条离线索引文件约 1.94 MiB；本地基准中解析后约保留 4.65 MiB、加载瞬时峰值约 15.27 MiB。完整快照中 1815 条画像通过 GitHub List Commits 元数据接口绑定真实提交 OID，Tree 证据覆盖率 98.96%。扫描器不调用可能携带文件 patch 的 Commit 详情接口；顾问也不会把 GitHub 文件树缓存到生产插件目录，最终仍以目标服务器实测为准。

## 画像生成（维护者）

市场元数据模式不需要凭据：

```powershell
python scripts/scan_market.py --mode market
```

完整模式需要一个被 GitHub 接受、只读公共仓库的临时 Token，通过进程环境传入：

```powershell
$env:GITHUB_TOKEN = '<临时令牌>'
python scripts/scan_market.py --mode github --workers 2 --min-interval 0.75
Remove-Item Env:GITHUB_TOKEN
```

扫描器先执行每插件至多两个请求的强制阶段（List Commits 元数据 + Tree），并按批次写入仅供续扫的
checkpoint；只有整批完成后才发布索引。SBOM 是剩余额度允许时执行的可选增强阶段。额度不足、
超时或断网不会把部分结果发布成完整索引。扫描器不调用 archive、contents、Commit 详情或
`git clone`，也不下载第三方插件源码。扫描后执行：

```powershell
python scripts/validate_index.py --minimum-profiles 1800 --minimum-github-ratio 0.95
python scripts/sign_index.py --private-key '<离线私钥路径>'
```

签名私钥不得提交、打包或上传到服务器。插件包只包含公钥。

## 历史聊天离线分析（管理员）

如需在首次启用前评估已有 AstrBot 对话，可在服务器上只读运行：

```bash
python3 scripts/analyze_astrbot_db.py --database /path/to/data_v4.db > aggregate-report.json
```

脚本以 SQLite `mode=ro` 打开数据库，只提取用户角色的显式文本字段并立即转换为总数、词频、
主题和候选插件；报告不含整条消息、身份列、会话 ID、明文群号或用户 ID，但达到阈值的高频
词/短语仍来自聊天内容，可能包含人名或专名，应只在本机最小授权保存。不同 AstrBot 历史表
可能存在重复，因此报告明确标记为粗略样本，不能替代开启统计后的按群长期观察。

## 评分和风险等级

完整算法、保守上限和百分位定义见 [docs/scoring-spec.md](docs/scoring-spec.md)；
静态资源等级、证据来源和局限见 [docs/resource-methodology.md](docs/resource-methodology.md)。

必须牢记：L0–L4 是容量风险区间，不是实测值。输入规模、并发、内存泄漏、外部 API 延迟和平台实现都可能改变实际占用。

当前发行版优先加载 `data/source_resource_index.json`：其中 1815 个插件由本地源码只读静态分析生成，另有 19 个无本地源码条目沿用元数据画像。逐插件 LLM 资源补判默认关闭；只有管理员明确开启 `enable_llm_fallback` 后才会作为低置信度补充。

## 测试

```powershell
python -m unittest discover -s tests -v
python -m compileall -q advisor main.py scripts tests
python scripts/validate_index.py --minimum-profiles 1800 --minimum-github-ratio 0.95
python scripts/validate_index.py data/source_resource_index.json --minimum-profiles 1800
```

部署、验证和回滚见 [docs/operations.md](docs/operations.md)，安全设计见
[docs/security.md](docs/security.md)。

## 发布边界

- 推荐结果不会自动安装插件。
- 大模型不能改变评分权重，也不能降低确定性规则给出的风险等级。
- 群聊大模型可从去身份化摘要中发现新需求，但必须引用输入特征 ID；新需求只由本地规则匹配市场描述和标签并受强度上限约束，推荐仍由固定评分完成。
- 无法确认的事实写入 `unknowns`，不会假装为轻量。
- 市场、GitHub 和插件描述均视为不可信输入，不作为系统指令执行。
