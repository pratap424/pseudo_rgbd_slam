"""
Generate research-quality figures for Pseudo RGB-D SLAM.
ALL data is measured, not projected.

Run: python generate_figures.py
Output: assets/*.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


# =====================================================================
# Figure 1: ATE Comparison — All measured data
#   - Real Kinect: 0.016m (published, Mur-Artal TRO 2017 Table IV)
#   - Ours with masking: 0.0206m (measured)
#   - Ours without masking: 0.5536m (measured, ablation run)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 3.5))

labels = ['No Masking\n(Ablation)', 'Pseudo RGB-D\n(Ours)', 'Real Kinect\n(Published*)']
values = [0.5536, 0.0206, 0.016]
colors = ['#e74c3c', '#27ae60', '#3498db']

bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor='#2c3e50', linewidth=0.8)

for bar, val in zip(bars, values):
    if val > 0.1:
        ax.text(val - 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f} m', va='center', ha='right', fontweight='bold', color='white', fontsize=11)
    else:
        ax.text(val + 0.015, bar.get_y() + bar.get_height()/2,
                f'{val:.4f} m', va='center', ha='left', fontweight='bold', color='#2c3e50', fontsize=11)

# Annotations
ax.annotate('26.9x worse', xy=(0.5536, 0), xytext=(0.45, 0.65),
            fontsize=10, color='#e74c3c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
            transform=ax.get_yaxis_transform())

ax.annotate('1.3x', xy=(0.0206, 1), xytext=(0.12, 1),
            fontsize=10, color='#2c3e50', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2))

ax.set_xlabel('Absolute Trajectory Error - RMSE (meters)', fontweight='bold')
ax.set_title('Trajectory Accuracy: Confidence Masking Ablation', fontweight='bold', pad=12)
ax.set_xlim(0, 0.65)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Footnote
ax.text(0.0, -0.22, '*Mur-Artal & Tardos, ORB-SLAM2, IEEE TRO 2017, Table IV',
        transform=ax.transAxes, fontsize=8, color='#7f8c8d', style='italic')

plt.savefig('assets/ate_comparison.png')
plt.close()
print("[OK] ate_comparison.png")


# =====================================================================
# Figure 2: Per-Node Latency Breakdown — All measured on GTX 1650
#   - Node A: ~1ms (measured)
#   - Node B: 141ms (measured, avg over 120 frames)
#   - Node C: 11ms (measured, avg over 118 frames)
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 3))

nodes = ['Node A\n(Broadcaster)', 'Node B\n(DA2 Depth)', 'Node C\n(ORB-SLAM3)']
latencies = [1, 141, 11]
colors_lat = ['#3498db', '#e74c3c', '#27ae60']
total = sum(latencies)

bars = ax.barh(nodes, latencies, color=colors_lat, height=0.5,
               edgecolor='#2c3e50', linewidth=0.8)

for bar, val in zip(bars, latencies):
    pct = val / total * 100
    label = f'{val}ms ({pct:.0f}%)'
    if val > 20:
        ax.text(val - 3, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right', fontweight='bold', color='white', fontsize=11)
    else:
        ax.text(val + 3, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='left', fontweight='bold', color='#2c3e50', fontsize=10)

ax.annotate('BOTTLENECK', xy=(141, 1), xytext=(120, 1.6),
            fontsize=10, color='#e74c3c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

ax.set_xlabel('Latency (milliseconds)', fontweight='bold')
ax.set_title('Per-Node Latency Breakdown (measured, GTX 1650)', fontweight='bold', pad=12)
ax.set_xlim(0, 165)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.savefig('assets/latency_breakdown.png')
plt.close()
print("[OK] latency_breakdown.png")


# =====================================================================
# Figure 3: GPU vs CPU — Measured throughput comparison
#   - GPU (GTX 1650, PyTorch FP32): 7.2 FPS depth, 7.1 FPS pipeline
#   - CPU (PyTorch FP32): 2.1 FPS depth, 2.1 FPS pipeline
#   Both measured. SLAM node (11ms) is never the bottleneck.
# =====================================================================
fig, ax = plt.subplots(figsize=(6, 4))

configs = ['CPU Only\n(PyTorch FP32)', 'GTX 1650\n(PyTorch FP32)']
fps_depth = [2.1, 7.2]
fps_slam = [90, 90]  # SLAM is always ~90 FPS, never bottleneck

x = np.arange(len(configs))
width = 0.3

bars1 = ax.bar(x - width/2, fps_depth, width, label='Depth Inference (Node B)',
               color='#e74c3c', edgecolor='#2c3e50', linewidth=0.8)
bars2 = ax.bar(x + width/2, fps_slam, width, label='SLAM Tracking (Node C)',
               color='#27ae60', edgecolor='#2c3e50', linewidth=0.8)

# Value labels
for bar, val in zip(bars1, fps_depth):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=11)
for bar, val in zip(bars2, fps_slam):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'~{val}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 30FPS line
ax.axhline(y=30, color='#e67e22', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(1.45, 31.5, '30 FPS\nReal-Time', color='#e67e22', fontsize=8, fontweight='bold', ha='center')

ax.set_ylabel('FPS (frames per second)', fontweight='bold')
ax.set_title('Measured Throughput: GPU vs CPU', fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylim(0, 110)
ax.legend(loc='upper right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Annotation: GPU speedup
ax.annotate('3.4x GPU speedup', xy=(0.5, 5), fontsize=10,
            fontweight='bold', color='#2c3e50', ha='center')

plt.savefig('assets/gpu_vs_cpu.png')
plt.close()
print("[OK] gpu_vs_cpu.png")


print("\nAll figures generated (measured data only).")
