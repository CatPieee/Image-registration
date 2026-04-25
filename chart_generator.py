import matplotlib.pyplot as plt
import numpy as np
import json
import os

def generate_charts():
    stats_path = "output/stats.json"
    if not os.path.exists(stats_path):
        print("未找到统计数据文件，请先运行 main.py")
        return

    with open(stats_path, "r") as f:
        data = json.load(f)

    # 提取数据
    sample_names = list(data.keys())
    methods = ["LoFTR", "SIFT"]
    
    loftr_inliers = [data[name]["LoFTR"]["inliers"] for name in sample_names]
    sift_inliers = [data[name]["SIFT"]["inliers"] for name in sample_names]

    # 1. 绘制所有样本的 Inliers 对比图
    x = np.arange(len(sample_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, loftr_inliers, width, label='LoFTR (Proposed)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, sift_inliers, width, label='SIFT (Baseline)', color='#c44e52')

    ax.set_ylabel('Number of RANSAC Inliers')
    ax.set_title('Cross-Modal Registration Robustness across Multiple TNO Sequences')
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, rotation=15)
    ax.legend()

    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("output/multi_sample_comparison.png", dpi=300)
    print("多样本对比图已保存至 output/multi_sample_comparison.png")

    # 2. 绘制平均性能对比图
    avg_loftr = np.mean(loftr_inliers)
    avg_sift = np.mean(sift_inliers)

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(methods, [avg_loftr, avg_sift], color=['#4c72b0', '#c44e52'], width=0.6)
    ax.set_ylabel('Average RANSAC Inliers')
    ax.set_title('Average Performance across Dataset')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("output/average_performance.png", dpi=300)
    print("平均性能图已保存至 output/average_performance.png")

if __name__ == "__main__":
    generate_charts()
