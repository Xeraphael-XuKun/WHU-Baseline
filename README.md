# WHU-Baseline

这是快照派生的 WHU-MARS baseline 项目。

## 来源与边界

- 原始源码导入提交：`78f80a2`，tag：`teacher-snapshot`。
- 老师历史结果仍只以原目录中的日志和 checkpoint 为证据；训练产物没有复制进本仓库。
- 当前没有清理老师快照中的非 baseline 模块。候选配置通过显式开关关闭这些模块。
- 官方 TransReID 是上游参考，不是本项目的可执行起点。

## 三组候选实验

| 任务 | 配置 | 采样 | K | 逻辑 batch | 实际图片 batch |
|---|---|---|---:|---:|---:|
| `A_baseline_candidate` | `configs/A_baseline_candidate.yml` | 老师普通 PKM，P=16 | 4 | 64 | 192 |
| `B_baseline_candidate` | `configs/B_baseline_candidate.yml` | P_AG=8、P_G=8 的分层 PKM | 4 | 64 | 192 |
| `C_fused_baseline_candidate` | `configs/C_fused_baseline_candidate.yml` | P_AG=8、P_G=8 的分层 PKM | 2 | 32 | 96 |

三组共同使用 raw OpenAI CLIP ViT-B/16、CLIP native normalization、单一
768D pre-projection CLS、CE + soft-margin triplet、Adam、两级学习率、60
epochs、100-update warmup、per-update cosine、DropPath 0→0.1、全局梯度裁剪
1.0，以及 `R1_mAP_eval(metric='sysu', FEAT_NORM='yes')`。

## 服务器任务

先单独执行：

```bash
bash /mnt/cache/wanghanzhi/XK/WHU-Baseline/server/preflight_a800.sh
```

随后在平台一次性创建三个独立任务，并按以下顺序串行调度：

```text
1. server/run_A_baseline_candidate_a800.sh
2. server/run_B_baseline_candidate_a800.sh
3. server/run_C_fused_baseline_candidate_a800.sh
```

每个脚本均使用 `CUDA_VISIBLE_DEVICES=0`、`WORLD_SIZE=1`，从同一预训练权重
重新初始化，写入独立输出目录，并在训练结束后从磁盘重新加载
`transformer_60.pth` 复评。

详细设置和证据边界见 `doc/0.三组baseline候选实验与服务器启动手册_0922.md`。

## A 与老师历史运行差异的 10 轮诊断

这部分代码只存在于 `diagnostic/env-cudnn-e10` 诊断分支，不属于正式
baseline。三组均使用同一单卡 A800、`SEED=1234` 和同一 A 配方，保留
60-epoch scheduler 地平线，但在 epoch 10 诊断停止：

| 任务 | Python 环境 | cuDNN benchmark | 主要问题 |
|---|---|---:|---|
| D10 | `whu_mars` | False | 当前 A 设置在相同环境下能否短程重复 |
| E10 | `whu_mars` | True | 只改变 benchmark 后轨迹是否移动 |
| F10 | `llmpar` | True | 在 E10 基础上再改变历史环境候选 |

先执行双环境预检，再在平台创建三个独立单卡任务并按 D10、E10、F10
串行调度：

```text
server/preflight_diagnostic_e10_a800.sh
server/run_D_A_repeat_benchmark_false_a800.sh
server/run_E_A_benchmark_true_a800.sh
server/run_F_teacher_env_benchmark_true_a800.sh
```

三组分别写入 `WHU-Baseline_runs/diagnostic_e10/` 下的独立目录，并从磁盘
加载 `transformer_10.pth` 统一复评。10 轮只用于定位早期轨迹差异，不能替代
完整 60 轮、固定 epoch-60 checkpoint 的正式性能比较。详细设置、判读规则和
代码治理见 `doc/3.cuDNN_benchmark最小对照实验_0922.md`。
