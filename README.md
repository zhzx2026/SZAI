# SZ-AI-R1/V1

这是一个可以直接放到 GitHub 上运行的轻量训练项目，用 GitHub Actions 训练一个字符级中文语言模型 `SZ-AI-R1/V1`。

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
