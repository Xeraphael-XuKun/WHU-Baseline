# WHU-Baseline：A → Trajectory

本分支只保留已选定的 A baseline 主训练路径，下一步研究核心为 **A + Trajectory**。
VTC 仅作为后续补充对照；本轮未实现 Trajectory，也未启用 VTC。

## 冻结设置

CLIP ViT-B/16，256×128，CLIP native normalization，普通 PKM（P16/K4，
三模态，逻辑 batch 64 / 实际图片 192），CE + soft-margin batch-hard Triplet，
Adam，主干 LR 5e-6 / 新参数 LR 3.5e-4，100 updates warmup + cosine，
60 epochs，seed 1234，benchmark=False。Triplet 和正式检索均使用 768D pre-BN。
评测为 SYSU 同相机过滤、L2 normalization、无 reranking，固定 transformer_60.pth。

## 当前入口

- 配置：`configs/A_baseline_candidate.yml`
- 训练 / 复评：`train.py` / `test.py`
- 服务器预检：`server/preflight_a800.sh`
- A 单卡前台训练及独立复评：`server/run_A_baseline_candidate_a800.sh`
- 清理范围与验收：`doc/5.A基线清理与Trajectory开发边界_0929.md`

现有 A 输出目录非空时启动脚本会拒绝覆盖。新的重跑应使用独立输出目录。
预检成功不等于正式训练已启动。

## 历史与兼容边界

清理前源码为 `cb2d34a`，A/B/C 实际训练的基础版本为 `d0b1f09`。
B/C、WRT/UAD、PLD、模态残差、Twin/TNCE、ChartPE/RoPE 等历史实现可从 Git 历史恢复，
对应旧文档仅描述历史实验，不是本分支的启动说明。原始老师项目不作修改。

为了保持 A 初始化和 checkpoint 兼容，保留未用于损失的 CenterLoss 构造及其 optimizer、
backbone 的闲置 fc 和 CLIP visual.proj。它们不构成附加方法。
`clip_text.py`、`clip_tokenizer.py`、`loss/text_align.py` 暂存为未来补充 VTC 的独立参考工具，
未被模型和训练器导入；其中旧的可学习 prompt 不等于未来冻结锚点 VTC。

训练产物、原始结果、idea 文档和未提交的用户文档均不纳入本次代码提交。
