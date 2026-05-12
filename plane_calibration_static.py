import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================================
# 1. 加载 plane calibration 参数
# =========================================
# 这个文件来自你的 plane calibration
# 内部应该包含:
#   R : 3x3 rotation matrix
#   t : 3x1 translation vector
#
# 与 plane_calibration_visualization.py 保持一致:
#   Pp = R.T @ (Pc - t)
# =========================================
plane = np.load("plane_calibration_flipped.npz")
R = plane["R"]
t = plane["t"]

# =========================================
# 2. 加载 CSV
# =========================================
csv_path = "aruco_rfid_20260511_233439.csv"

# 如果你的文件名带 (1)
# csv_path = "aruco_rfid_20260511_233439(1).csv"

df = pd.read_csv(csv_path)

# =========================================
# 3. 提取 HAND 轨迹
# =========================================
# 原始相机坐标
# Pc = [x, y, z]
# =========================================
hand_xyz = df[["HAND_x", "HAND_y", "HAND_z"]].values

# 去除 NaN
mask = ~np.isnan(hand_xyz).any(axis=1)
hand_xyz = hand_xyz[mask]

# =========================================
# 4. 梯形/平面校正
# =========================================
# 使用与 plane_calibration_visualization.py
# 完全相同的变换:
#
# Pp = R.T @ (Pc - t)
#
# 作用:
#   - 消除相机视角导致的梯形畸变
#   - 转到平面坐标系
#   - Z 轴表示偏离平面的高度
# =========================================
corrected_points = []

for Pc in hand_xyz:
    Pp = R.T @ (Pc - t)
    corrected_points.append(Pp)

corrected_points = np.array(corrected_points)

# =========================================
# 5. 2D Trace 可视化
# =========================================
# 使用 plane 坐标系中的 XY
# =========================================
plt.figure(figsize=(8, 8))

plt.plot(
    corrected_points[:, 0],
    corrected_points[:, 1],
    linewidth=2,
    label="Corrected Hand Trace"
)

# 起点
plt.scatter(
    corrected_points[0, 0],
    corrected_points[0, 1],
    s=100,
    marker='o',
    label="Start"
)

# 终点
plt.scatter(
    corrected_points[-1, 0],
    corrected_points[-1, 1],
    s=100,
    marker='x',
    label="End"
)

plt.xlabel("Plane X")
plt.ylabel("Plane Y")
plt.title("HAND Trace After Plane / Trapezoid Correction")
plt.axis('equal')
plt.grid(True)
plt.legend()

# =========================================
# 6. 3D 可视化
# =========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    corrected_points[:, 0],
    corrected_points[:, 1],
    corrected_points[:, 2],
    linewidth=2
)

# 起点
ax.scatter(
    corrected_points[0, 0],
    corrected_points[0, 1],
    corrected_points[0, 2],
    s=100,
    label='Start'
)

# 终点
ax.scatter(
    corrected_points[-1, 0],
    corrected_points[-1, 1],
    corrected_points[-1, 2],
    s=100,
    label='End'
)

ax.set_xlabel("Plane X")
ax.set_ylabel("Plane Y")
ax.set_zlabel("Height from Plane")
ax.set_title("3D Corrected HAND Trace")

# 保持比例一致
max_range = np.array([
    corrected_points[:, 0].max() - corrected_points[:, 0].min(),
    corrected_points[:, 1].max() - corrected_points[:, 1].min(),
    corrected_points[:, 2].max() - corrected_points[:, 2].min()
]).max() / 2.0

mid_x = (corrected_points[:, 0].max() + corrected_points[:, 0].min()) * 0.5
mid_y = (corrected_points[:, 1].max() + corrected_points[:, 1].min()) * 0.5
mid_z = (corrected_points[:, 2].max() + corrected_points[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.legend()

plt.show()