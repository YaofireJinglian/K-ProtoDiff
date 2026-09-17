# 缺少标准差的基线补跑

按用户 2026-09-09 要求，将原来仅引用论文单值的基线扩展为独立三种子本地复现。

- 9 数据集 × 7 方法 × 3 种子 = 189 个训练/采样/全指标任务；加上 fMRI GT-GAN 两个失败种子，共 191 个任务。Traffic GT-GAN 在论文中为 OOM，本次仍尝试，失败就如实记录，不伪造数值。
- 固定种子 2026/2027/2028，不筛选有利种子。每个数据集配置在 `Config/baselines/<dataset>.yaml`。
- 使用对应期刊正式实验已保存的归一化真实数组，保持输入样本相同。原论文九个数据集的全部独立训练产物不在本工作区，因此不能只重评估现有单个模型就声称获得三训练种子方差。
- 四个非扩散方法沿用已跑通的官方 fMRI 适配器及固定预算。PaD-TS 迁移 fMRI 配置，按数据改变输入维度；Diffusion-TS 使用各数据集已有规模和训练预算；会议 K-ProtoDiff 明确关闭多尺度/自适应采样，并采用 legacy FDM、point 原型。
- 这是透明记录配置的本地复现，不是恢复原论文的精确超参和指标实现。未调参保证新基线与论文均值一致；也不将新标准差拼接到旧均值。
- 全指标使用 `eval_seeded.py`：C-FID、KL、DS、PS、DTW-6/8/10。结果汇总为均值 ± 样本标准差，三位小数。
- 新记录齐三种子后替换相应论文均值与标准差，用 † 标识。本次实际复现和旧论文引用混合阶段，不作跨来源排名。
- 新产物位于 `checkpoints/baseline_reproduction/<dataset>/<method>/seed<seed>`。旧 fMRI GT-GAN 权重不覆盖：两个缺失种子从头训练，因为旧产物不具备完整优化器续训状态。
- 持久 watchdog 优先安排缺项补跑，不中断在跑搜索。CUDA 初始化失败时禁止启动新任务；失败任务保留日志并标记待检查，不反复从头重训。
- `BASELINE_UNCERTAINTY_STATUS.md` 是实际队列状态，pending 不代表已启动。正式任务与 `OUTPUT/baseline_reproduction_smoke` 的短测试完全隔离。

启动验证：Stocks TimeVAE 的 GPU 一轮训练与生成已通过。验证扩散适配器时，GPU 0 再次出现 NVML Unknown Error，其余抽测 GPU 的原生 cuInit 返回 999，未能完成 GPU 验证。没有将短测试计入正式结果。
