# SPIM V7 + RL Mainline — Windows 部署指南

## 1. 项目简介

本项目为 SPIM V7 + RL (Set-level Policy Imitation via Soft
Posterior + Reinforcement Learning) 的主线代码。核心功能：

- **SPIM 教师策略评估** — 基于 soft scenario posterior 的多教师家族推理
- **RL 策略训练** — Actor-Critic 强化学习模仿教师策略
- **论文分析工具** — SPIM v3/v6/v7 论文图渲染和审计

最优教师配置: `hsr_soft_scenario_posterior_v7_7offset, alpha=0.55`

## 2. 环境要求

- **操作系统**: Windows 10/11 x64
- **Python**: 3.10 或更高
- **GPU**: NVIDIA GPU + CUDA 12.x（推荐，CPU 也可运行但慢）
- **磁盘空间**: 至少 15 GB 空闲空间（代码 ~100MB + 数据 ~10GB）

## 3. 安装步骤

### 3.1 克隆代码

```powershell
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>
```

### 3.2 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3.3 安装 PyTorch Geometric（额外步骤）

`torch-geometric` 需要分步安装，请根据你的 CUDA 版本选择：

```powershell
# 先确认 PyTorch 版本
python -c "import torch; print(torch.__version__)"

# 然后安装 torch-geometric（以 CUDA 12.1 为例）
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install torch-geometric
```

如果没有 GPU，用 CPU 版：
```powershell
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install torch-geometric
```

## 4. 数据部署（⚠️ 关键步骤）

代码不包含大型数据文件。你需要从源服务器获取以下数据并放置在正确位置。

### 4.1 目录结构总览

```
<项目根目录>/
├── src/                     ← Git 已包含
├── tools/                   ← Git 已包含
├── scripts/                 ← Git 已包含
├── data/                    ← 需要你手动准备 ⬇
│   ├── train.txt            ← Git 已包含 (1.7MB)
│   ├── val.txt              ← Git 已包含 (356KB)
│   ├── test.txt             ← Git 已包含 (352KB)
│   └── cache_lmdb/          ← 需要从源服务器拷贝 (~428MB)
│       ├── v6_dataset_train_setrl_main_train4823_full_seed46_v1.lmdb/
│       ├── v6_dataset_train_setrl_main_train4823_full_seed45_v1.lmdb/
│       ├── v6_dataset_train_setrl_main_train256_v1.lmdb/
│       ├── v6_dataset_val_setrl_main_train256_v1.lmdb/
│       ├── v6_dataset_test_setrl_main_train256_v1.lmdb/
│       ├── v6_dataset_train_clean_aligned_online_finish_train_full_e0c82c54.lmdb/
│       ├── v6_dataset_val_clean_aligned_online_finish_train_full_e0c82c54.lmdb/
│       ├── v6_dataset_test_clean_aligned_online_finish_train_full_e0c82c54.lmdb/
│       └── v6_dataset_train_train_full_rlpilot_smoke_n4_v1.lmdb/
└── datanew/                 ← 需要从源服务器拷贝 (~8.9GB)
    └── production_data/
        └── foundation_20260114_164946_86d5023e/
            ├── graph.npz
            ├── metadata.json
            ├── active_nodes_list.json
            └── subgraph_v11_prod/
                ├── event_*.npz   (数千个文件)
                └── ...
```

### 4.2 从源服务器获取数据

在源服务器（Linux）上，执行以下命令打包数据：

```bash
# 在源服务器 rl_spim_v7_mainline 目录下执行

# 打包 data/cache_lmdb（约 400MB 压缩后）
tar -czf cache_lmdb.tar.gz -C /root/autodl-tmp/rl_spim_v7_mainline data/cache_lmdb

# 打包 datanew/（约 8GB 压缩后，耗时较长）
tar -czf datanew.tar.gz -C /root/autodl-tmp/rl_spim_v7_mainline datanew
```

然后将这两个 `.tar.gz` 文件传输到 Windows 机器（百度网盘/U盘/scp 均可）。

### 4.3 在 Windows 上解压数据

```powershell
# 在项目根目录执行

# 解压 cache_lmdb（约 428MB）
tar -xzf cache_lmdb.tar.gz

# 解压 datanew（约 8.9GB，耗时较长）
tar -xzf datanew.tar.gz
```

### 4.4 验证数据完整性

```powershell
python -c "
import os
# 检查关键文件
files = [
    'data/train.txt',
    'data/val.txt', 
    'data/test.txt',
    'datanew/production_data/foundation_20260114_164946_86d5023e/graph.npz',
    'datanew/production_data/foundation_20260114_164946_86d5023e/metadata.json',
]
for f in files:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f'  {\"✓\" if exists else \"✗\"} {f} ({size/1024:.0f} KB)')

# 检查 LMDB 缓存
import glob
lmdb_dirs = glob.glob('data/cache_lmdb/v6_dataset_*')
print(f'\\nLMDB 缓存目录: {len(lmdb_dirs)} 个')
"
```

## 5. 运行验证

### 5.1 冒烟测试

以下命令使用 1 个 case 验证环境是否完整（在项目根目录执行）：

```powershell
# 测试 V7 教师策略评估（CPU 模式，1 case）
python -m src.scripts.run_spim_policy_eval_strict `
  --teacher-family hsr_soft_scenario_posterior_v7_7offset `
  --paper-like-alpha 0.55 `
  --policy-mode teacher `
  --policy-name smoke_v7_teacher `
  --output-dir artifacts/smoke_runs/strict_v7_teacher_val1 `
  --split val `
  --case-limit 1 `
  --trace-case-limit 1 `
  --trace-step-limit 1 `
  --device cpu
```

> **注意**: Windows PowerShell 中换行符是 `` ` ``（反引号），不是 Linux 的 `\`。

### 5.2 小规模 RL 训练测试

```powershell
python -m src.scripts.run_spim_teacher_imitation_rl_pilot `
  --teacher-family hsr_soft_scenario_posterior_v7_7offset `
  --paper-like-alpha 0.55 `
  --runner-version-tag smoke_v7_7offset_seed45 `
  --output-dir artifacts/smoke_runs/train_v7_seed45_n4 `
  --train-full-max-cases 4 `
  --train-full-cache-version train_full_rlpilot_smoke_n4_v1 `
  --bc-epochs 1 `
  --bc-recovery-epochs 0 `
  --rl-epochs 1 `
  --rl-update-epochs 1 `
  --device cpu `
  --save-final-checkpoint artifacts/smoke_runs/train_v7_seed45_n4/checkpoints/rl_student_final.pt
```

### 5.3 如果有 GPU

将 `--device cpu` 改为 `--device cuda` 即可启用 GPU 加速。

## 6. 常见问题

### 6.1 `ModuleNotFoundError: No module named 'src'`

原因：没有在项目根目录运行。请确保 `cd` 到项目根目录（包含 `src/` 文件夹的那一级）。

### 6.2 `FileNotFoundError: datanew/production_data/...`

原因：数据没有放置到位。请按照第 4 节检查数据目录结构。

### 6.3 `ImportError: libcrypto.so` 或类似 Linux 库错误

本项目代码是跨平台的（使用 `pathlib.Path`），但某些底层依赖
（如 `lmdb`）需要在 Windows 上重新编译。确保使用 `pip install lmdb`
安装，`pip` 会自动处理 Windows 兼容性。

### 6.4 多进程相关警告

Windows 的 `multiprocessing` 使用 `spawn` 方式（与 Linux 的 `fork` 不同），
代码中已做了兼容处理。如果看到 `if __name__ == '__main__'` 相关的报错，
通常是因为在交互式环境中运行了脚本——请确保通过 `python -m` 命令运行。

## 7. 项目入口速查

| 功能 | 命令 |
|------|------|
| V7 教师评估 | `python -m src.scripts.run_spim_policy_eval_strict` |
| RL 训练 | `python -m src.scripts.run_spim_teacher_imitation_rl_pilot` |
| 教师家族扫描 | `python -m src.scripts.run_spim_family_sweep` |
| 论文图渲染 | `python tools/render_spim_semantic_publication_figures.py` |
| 基础策略评估 | `python -m src.scripts.run_spim_policy_eval` |
