#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt

# ====== 路径配置 ======
EVAL_DIR = Path("/mnt/sh/mmvision/home/jonahli/save/agent/eval/im2gps3k")
OUT_PNG = Path("/mnt/sh/mmvision/home/jonahli/projects/tusou/script/sft/sft_4b_scaling_curves.png")

# ====== 步数范围 ======
steps = list(range(200, 5601, 200))  # 200, 400, ..., 5600

# 每200 step 对应10张训练数据
# 训练数据量 = step / 200 * 10 = step / 20
train_images = [s // 20 for s in steps]

metrics = {
    "acc_1km": [],
    "acc_25km": [],
    "acc_200km": [],
    "acc_750km": [],
    "acc_2500km": [],
}

valid_steps = []
valid_train_images = []

for s in steps:
    f = EVAL_DIR / f"sft_4b_step{s}_v2_summary.json"
    if not f.exists():
        print(f"[WARN] 缺文件，跳过: {f}")
        continue

    with open(f, "r", encoding="utf-8") as rf:
        d = json.load(rf)

    # 如果某个 key 缺失就报错，避免 silently wrong
    for k in metrics:
        if k not in d:
            raise KeyError(f"{f} 缺少字段: {k}")

    valid_steps.append(s)
    valid_train_images.append(s // 20)
    for k in metrics:
        metrics[k].append(d[k])

# ====== 作图 ======
plt.figure(figsize=(6, 6), dpi=150)

for k, y in metrics.items():
    plt.plot(valid_train_images, y, marker="o", linewidth=1.8, markersize=3.5, label=k)

plt.title("SFT 4B Scaling Curves on im2gps3k")
plt.xlabel("Training images x10000")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.0)
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.tight_layout()

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG)
print(f"[OK] 图已保存: {OUT_PNG}")

# 可选：打印最后一个点方便核对
if valid_steps:
    print(f"[INFO] 最后step={valid_steps[-1]}, train_images={valid_train_images[-1]}")
    for k in metrics:
        print(f"  {k}: {metrics[k][-1]:.4f}")
