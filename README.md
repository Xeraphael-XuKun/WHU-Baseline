# WHU-Baseline

这是从老师当前有效 HiHR/TransReID 快照派生的 WHU-MARS baseline 项目。

## 来源与边界

- 原始证据目录：`E:\CVPR2027\ref\HiHR`，保持不修改。
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
