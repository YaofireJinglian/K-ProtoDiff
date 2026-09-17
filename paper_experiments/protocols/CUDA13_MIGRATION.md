# CUDA 13 环境迁移

用户于 2026-09-09 要求升级项目 CUDA 运行环境。

- 新环境：`.venv-cu130`，Python 3.11，PyTorch 2.9.1+cu130，torchvision 0.24.1+cu130。
- 版本组合来自 PyTorch 官方：https://pytorch.org/get-started/previous-versions/#v291 。
- 重建命令：`bash scripts/setup_cu130.sh`。安装源仅 PyPI 与 PyTorch 官方 cu130 索引；固定版本在 `requirements-cu130.txt`。
- 本次升级的是 PyTorch wheel 随附的 CUDA 13.0 运行库，不安装系统级 Toolkit/nvcc，也不改 NVIDIA 驱动。
- `.venv`（PyTorch 2.7.1+cu128）和 `.venv-tf` 保留。TensorFlow 基线不迁移；正式基线共享评测继续使用原 `.venv`，避免同时改变评测环境。
- 自动调度的 PyTorch 解释器由 `Config/journal_runtime.yaml` 的 `pytorch_python` 指定。未设置时仍用 `.venv/bin/python`。
- 旧进程不会热切换；后续新进程才使用选定解释器。CUDA 健康检查失败时仍禁止新任务启动。
- 回退：将 `pytorch_python` 改为 `.venv/bin/python`。不用删除任何 checkpoint 或安装目录。
- 注意：变更 PyTorch/CUDA 可能改变浮点数与随机过程行为，跨版本续训不应宣称与原环境逐位复现。已完成结果保持不变。

## 验证结果（2026-09-09）

- 安装完成，`torch.__version__ = 2.9.1+cu130`，`torch.version.cuda = 13.0`。
- `pip check` 通过；torchvision 与 TensorFlow CPU 导入通过；已安装完整版本快照在 `requirements-cu130-lock.txt`。
- 新环境运行项目测试：21 passed；测试在 CPU 上完成，伴随 CUDA 初始化失败警告，不是 GPU 实测通过。
- 原 fMRI seed 2032/lr_prototype 的 step 1500 checkpoint 严格加载通过，CPU 前向/反向通过，loss 有限。
- 新环境 `torch.cuda.is_available()` 仍为 False，CUDA 初始化依然失败。升级没有解决当前驱动/GPU 故障；CUDA 计算与多卡稳定性仍待服务器恢复后验证。
- `pytorch_python` 已指向 `.venv-cu130/bin/python`，仅影响后续调度的新 PyTorch 任务；健康检查仍阻止故障状态下启动。
- 安装器原下载完成后，停止了两个冗余的分段下载进程；未删除旧环境、实验数据或 checkpoint。
