# SZ-AI-R1/V1

这是一个可以直接放到 GitHub 上运行的轻量训练项目，用 GitHub Actions 训练一个字符级中文语言模型 `SZ-AI-R1/V1`。

如果你想先看整个项目现在的完整实现逻辑，可以直接看 `docs/AI_LOGIC.md`。

这个方案的目标不是在 GitHub Hosted Runner 上硬跑大模型，而是先把以下几件事打通：

- 有一条能复现的训练流水线
- 有一个能实际收敛的轻量模型
- 有可下载的模型产物
- 有训练后自动生成的样例文本

## 目录

```text
.
├── .github/workflows/train-sz-ai-r1-v1.yml
├── configs/sz-ai-r1-v1.json
├── data/train.txt
├── requirements.txt
├── scripts/generate.py
├── scripts/train.py
└── src/sz_ai/
```

## 模型说明

- 模型类型：字符级 GRU 语言模型
- 默认用途：训练一个轻量中文文本生成模型
- 适用环境：GitHub Hosted CPU Runner
- 不适用场景：大语言模型预训练、大规模 LoRA 微调、长时间 GPU 训练

如果你后面要训练更大的 `SZ-AI-R1/V1`，建议把这个 Action 改成：

1. 继续由 GitHub Actions 触发
2. 但把 `runner_labels` 改成 self-hosted GPU runner
3. 或者由 Action 去调用云端 GPU 训练任务

## 数据格式

默认训练语料是 `data/train.txt`。

你可以直接把自己的语料替换进去，建议格式是纯文本，内容尽量接近你希望模型学到的风格，例如：

```text
用户: 你好
SZ-AI: 你好，我是 SZ-AI-R1/V1。

用户: 帮我写一个 Python 脚本
SZ-AI: 先明确输入输出，再实现脚本。
```

语料越贴近最终用途，效果越稳定。

## 本地训练

```bash
python3 -m pip install -r requirements.txt
python3 scripts/train.py --config configs/sz-ai-r1-v1.json
python3 scripts/generate.py --checkpoint artifacts/SZ-AI-R1-V1/model.pt --prompt "SZ-AI: "
```

如果你在 macOS 本地运行时遇到 `OMP: Error #179`，可以临时这样执行：

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 scripts/train.py --config configs/sz-ai-r1-v1.json
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 scripts/generate.py --checkpoint artifacts/SZ-AI-R1-V1/model.pt --prompt "SZ-AI: "
```

训练产物会输出到 `artifacts/SZ-AI-R1-V1/`，包括：

- `model.pt`：模型权重和配置
- `metrics.json`：训练指标
- `summary.json`：训练摘要
- `sample.txt`：训练结束后的样例文本
- `generated.txt`：额外生成的文本

## GitHub Actions 训练

先把这个目录推到 GitHub 仓库，然后在 GitHub 上运行工作流：

1. 打开仓库的 `Actions`
2. 选择 `Train SZ-AI-R1-V1`
3. 点击 `Run workflow`
4. 按需调整这些输入项：

- `dataset_path`：训练语料路径
- `epochs`：训练轮数
- `batch_size`：批大小
- `output_dir`：产物目录
- `artifact_name`：上传后的 artifact 名称
- `runner_labels`：Runner 标签，默认是 `["ubuntu-latest"]`

训练完成后，你可以在 Actions 的 Artifacts 中下载模型产物。

## 代码助手版本

如果你的目标是做一个偏代码方向的 `SZ-AI`，更现实的路线不是硬追 DeepSeek 规模，而是：

- 只学习热门开源代码仓库
- 自动过滤许可证和非代码仓库
- 自动跳过 `awesome-*`、curated list 这类清单仓库
- 只提取代码文件，不把整个 GitHub 生吞进训练集
- 控制语料总量，让免费 GitHub Hosted Runner 也能训完

对应工作流是 `./.github/workflows/train-sz-ai-code-r1-v1.yml`。

它会做三件事：

1. 从 GitHub 实时搜索高 star 开源代码仓库
2. 浅克隆并提取代码文件，生成 `build/code-corpus/train.txt`
3. 训练一个代码方向的小型 `SZ-AI-Code-R1/V1`

默认参数偏保守：

- 语言：`C++,Python,Java`
- 总仓库数：`18`
- 每种语言最多：`6`
- 最低 star：`10000`
- 总文件上限：`480`
- 总语料字节上限：`1800000`
- 许可证白名单：`MIT, Apache-2.0, BSD-3-Clause, BSD-2-Clause, ISC, MPL-2.0`
- 严格语言文件过滤：只保留 `C++ / Python / Java` 对应源码文件

这样做不是因为技术上不能更多，而是因为免费 Action 的 CPU、磁盘、时长都有限。先做小而稳，才有可重复性。

如果你关心“训练器是不是要改成 C++ 才更快”，当前答案是不值得。对免费 GitHub Hosted Runner 来说，真正的瓶颈是：

- CPU 算力
- 模型结构
- 训练数据规模

所以当前更有效的优化方向不是把训练脚本重写成 C++，而是把训练语料收窄到你真正关心的语言，并把 repo 量和语料上限调到更合理的范围。现在默认就是按这个思路配的。

如果你要手动运行这个代码助手训练：

1. 打开仓库的 `Actions`
2. 选择 `Train SZ-AI-Code-R1-V1`
3. 点击 `Run workflow`
4. 按需调大 `repo_limit`、`per_language_limit` 或 `epochs`

这条工作流的 artifact 会额外包含：

- `selected-repos.json`：本次真正使用了哪些仓库
- `rejected-repos.json`：哪些候选被过滤掉以及原因
- `corpus-summary.json`：语料大小和提取统计

## macOS 应用

如果你想在 Mac 上直接点开一个本地程序来测试模型，现在仓库里已经有两个入口：

- `SZ-AI-Mac.command`
- `./.github/workflows/build-sz-ai-mac-app.yml`

先说清楚一点：macOS 不是 Windows，所以最终产物不会是 `.exe`，而是 `.app`。使用体验上就是双击打开。

### 直接在本机双击运行

先安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

然后双击仓库根目录里的 `SZ-AI-Mac.command`，会打开一个图形界面。你可以：

- 先点击 `Browse` 选择一个已经训练好的 `model.pt`
- 选择 `model.pt`
- 填 prompt
- 调整 `max_new_tokens`、`temperature`、`top_k`
- 点击 `Generate`

这个应用本身不内置模型权重，它只是一个本地测试器，所以必须先准备好训练产物里的 `model.pt`。

### 打包成真正的 `.app`

如果你想要一个更像发布版本的 Mac 应用，可以本地执行：

```bash
bash scripts/build_macos_app.sh
```

它会输出：

- `dist/SZ-AI-Mac.app`
- `dist/SZ-AI-Mac.zip`

### 用 GitHub Actions 构建 Mac 应用

如果你不想在本机装打包工具，可以在 GitHub 上运行工作流：

1. 打开仓库的 `Actions`
2. 选择 `Build SZ-AI Mac App`
3. 点击 `Run workflow`
4. 等待 artifact `sz-ai-mac-app`

下载后你会拿到已经打好的 `.app` 和 `.zip`。

## Self-hosted GPU 示例

如果你已经有自建 Runner，可以在运行工作流时把 `runner_labels` 改成：

```json
["self-hosted", "linux", "gpu"]
```

这样仍然由 GitHub Actions 编排，但实际训练会在你的 GPU Runner 上执行。

## 下一步建议

如果你要把它往真正可用的 `SZ-AI-R1/V1` 继续推进，下一步通常是：

1. 换成你自己的真实训练语料
2. 扩大语料规模并做清洗
3. 增加验证集和自动评估
4. 改成 GPU 训练
5. 再决定是否切到 Transformer 或 LoRA 微调路线
