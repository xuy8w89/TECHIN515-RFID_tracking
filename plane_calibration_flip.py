import numpy as np

# ==========================
# 读取原标定
# ==========================
plane = np.load("plane_calibration.npz")

R = plane["R"]
t = plane["t"]

print("Original R:")
print(R)

# ==========================
# 翻转 z 轴
# 同时翻转 x 轴保持右手系
# ==========================

R_new = R.copy()

# flip x
R_new[:, 0] *= -1

# flip z
R_new[:, 2] *= -1

print("\nModified R:")
print(R_new)

# ==========================
# 保存
# ==========================
np.savez("plane_calibration_flipped.npz", R=R_new, t=t)

print("\n✅ Saved: plane_calibration_flipped.npz")