import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2


# =========================================================
# Plane Calibration
# =========================================================
plane = np.load("plane_calibration.npz")

R_plane = plane["R"]
t_plane = plane["t"]


# =========================================================
# Camera -> Plane Coordinate
# 与 plane_calibration_visualization.py 保持一致
# =========================================================
def camera_to_plane(Pc):
    """
    Pc: [3]
        camera coordinate

    return:
        [3]
        plane calibrated coordinate
    """
    Pc = np.asarray(Pc)
    return R_plane.T @ (Pc - t_plane)


# =========================================================
# rvec -> arm direction
# 使用 ArUco marker 的 local z-axis
# =========================================================
def rvec_to_direction(rvec):

    Rm, _ = cv2.Rodrigues(rvec)

    # marker local z axis
    direction = Rm[:, 2]

    direction = direction / np.linalg.norm(direction)

    return direction


# =========================================================
# Smooth Arm Heatmap
# =========================================================
def build_arm_heatmap_smooth(
    wrist_point,
    arm_direction,
    grid_x,
    grid_y,
    heatmap_size=(128, 128),
    arm_length=0.35,
    arm_width=0.03,
):
    """
    Parameters
    ----------
    wrist_point:
        [3]
        calibrated plane coordinate

    arm_direction:
        [3]
        normalized direction vector

    grid_x:
        (xmin, xmax)

    grid_y:
        (ymin, ymax)

    heatmap_size:
        output resolution

    arm_length:
        arm ray length in meters

    arm_width:
        gaussian tube width

    Returns
    -------
    heatmap:
        [H,W]
        value = z-height weighted occupancy
    """

    H, W = heatmap_size

    xmin, xmax = grid_x
    ymin, ymax = grid_y

    # =====================================================
    # Arm segment
    # =====================================================
    p0 = wrist_point
    p1 = wrist_point + arm_direction * arm_length

    # =====================================================
    # XY grid
    # =====================================================
    xs = np.linspace(xmin, xmax, W)
    ys = np.linspace(ymin, ymax, H)

    X, Y = np.meshgrid(xs, ys)

    # =====================================================
    # Heatmap plane points
    # z=0 only for distance computation
    # =====================================================
    P = np.stack([
        X,
        Y,
        np.zeros_like(X)
    ], axis=-1)

    # =====================================================
    # Segment direction
    # =====================================================
    seg = p1 - p0
    seg_len2 = np.dot(seg, seg)

    # vector from p0 to every pixel
    v = P - p0

    # projection coefficient
    t = np.sum(v * seg, axis=-1) / seg_len2

    # clamp to segment
    t = np.clip(t, 0.0, 1.0)

    # =====================================================
    # Closest points on arm
    # =====================================================
    closest = p0 + t[..., None] * seg

    # =====================================================
    # Distance to arm
    # =====================================================
    dist = np.linalg.norm(P - closest, axis=-1)

    # =====================================================
    # Gaussian tube
    # =====================================================
    G = np.exp(
        -(dist ** 2) / (2 * arm_width ** 2)
    )

    # =====================================================
    # Z value
    # =====================================================
    z = closest[..., 2]

    # =====================================================
    # Final heatmap
    # =====================================================
    heatmap = z * G

    return heatmap


# =========================================================
# Recover 3D points from heatmap
# 可逆 representation
# =========================================================
def recover_points_from_heatmap(
    heatmap,
    grid_x,
    grid_y,
    threshold=1e-3,
):

    H, W = heatmap.shape

    xmin, xmax = grid_x
    ymin, ymax = grid_y

    ys, xs = np.where(np.abs(heatmap) > threshold)

    points = []

    for py, px in zip(ys, xs):

        x = xmin + (px / (W - 1)) * (xmax - xmin)
        y = ymin + (py / (H - 1)) * (ymax - ymin)

        z = heatmap[py, px]

        points.append([x, y, z])

    return np.array(points)


# =========================================================
# Load CSV
# =========================================================
CSV_PATH = "aruco_rfid_20260509_231938.csv"

df = pd.read_csv(CSV_PATH)


# =========================================================
# Parse all frames
# =========================================================
plane_points = []
arm_dirs = []

for _, row in df.iterrows():

    if pd.isna(row.get("WRIST_x")):
        continue

    # =====================================================
    # wrist position
    # =====================================================
    Pc = np.array([
        row["WRIST_x"],
        row["WRIST_y"],
        row["WRIST_z"],
    ])

    # camera -> plane
    Pp = camera_to_plane(Pc)

    # =====================================================
    # arm direction
    # =====================================================
    rvec = np.array([
        row["WRIST_rx"],
        row["WRIST_ry"],
        row["WRIST_rz"],
    ])

    d = rvec_to_direction(rvec)

    # direction also transform to plane coord
    d = R_plane.T @ d
    d = d / np.linalg.norm(d)

    plane_points.append(Pp)
    arm_dirs.append(d)

plane_points = np.array(plane_points)
arm_dirs = np.array(arm_dirs)

print("Loaded frames:", len(plane_points))


# =========================================================
# Global XY range
# 使用整个 CSV calibrated 后的范围
# =========================================================
xmin = plane_points[:, 0].min()
xmax = plane_points[:, 0].max()

ymin = plane_points[:, 1].min()
ymax = plane_points[:, 1].max()

# padding
pad = 0.05

xmin -= pad
xmax += pad

ymin -= pad
ymax += pad

grid_x = (xmin, xmax)
grid_y = (ymin, ymax)

print("Grid X:", grid_x)
print("Grid Y:", grid_y)


# =========================================================
# Visualization
# =========================================================
fig, ax = plt.subplots(figsize=(7, 7))

heatmap = build_arm_heatmap_smooth(
    plane_points[0],
    arm_dirs[0],
    grid_x,
    grid_y,
    heatmap_size=(128, 128),
    arm_length=0.35,
    arm_width=0.03,
)

im = ax.imshow(
    heatmap,
    origin="lower",
    cmap="jet",
    extent=[
        xmin,
        xmax,
        ymin,
        ymax
    ],
    vmin=-0.2,
    vmax=0.2,
)

plt.colorbar(im, ax=ax, label="Height Z")

ax.set_xlabel("Plane X")
ax.set_ylabel("Plane Y")

title = ax.set_title("Frame 0")


# =========================================================
# Animation update
# =========================================================
def update(frame_idx):

    heatmap = build_arm_heatmap_smooth(
        plane_points[frame_idx],
        arm_dirs[frame_idx],
        grid_x,
        grid_y,
        heatmap_size=(128, 128),
        arm_length=0.35,
        arm_width=0.03,
    )

    im.set_data(heatmap)

    title.set_text(f"Frame {frame_idx}")

    return [im]


# =========================================================
# Animation
# =========================================================
ani = FuncAnimation(
    fig,
    update,
    frames=len(plane_points),
    interval=50,
    blit=True,
)

plt.show()