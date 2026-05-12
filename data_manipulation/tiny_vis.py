import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.animation import FuncAnimation


# =========================================================
# Plane calibration
# 与 plane_calibration_visualization.py 保持一致
# =========================================================

def load_plane_calibration(path="plane_calibration.npz"):
    plane = np.load(path)
    R = plane["R"]
    t = plane["t"]
    return R, t


def camera_to_plane(Pc, R, t):
    """
    Pc: (...,3)

    plane_calibration_visualization.py 中：
        Pp = R.T @ (Pc - t)
    """

    return (R.T @ (Pc - t).T).T


# =========================================================
# Heatmap Representation
# =========================================================

def gaussian_weight(distance, sigma):
    """
    Gaussian falloff.

    distance=0 时权重=1
    距离越远逐渐衰减到 0

    sigma:
        控制边缘过渡宽度
    """

    return np.exp(-(distance ** 2) / (2 * sigma ** 2))


class ArmHeatmapEncoder:

    def __init__(
        self,
        x_range,
        y_range,
        resolution=128,
        max_ray_length=None,
        width=0.03,
    ):
        """
        x_range: (xmin, xmax)
        y_range: (ymin, ymax)

        resolution:
            heatmap resolution

        max_ray_length:
            射线最大长度。

            None:
                自动延伸到 heatmap 边界

        width:
            smooth width（米）
        """

        self.xmin, self.xmax = x_range
        self.ymin, self.ymax = y_range

        self.resolution = resolution
        self.max_ray_length = max_ray_length
        self.width = width

        self.dx = (self.xmax - self.xmin) / resolution
        self.dy = (self.ymax - self.ymin) / resolution

        xs = np.linspace(self.xmin, self.xmax, resolution)
        ys = np.linspace(self.ymin, self.ymax, resolution)

        self.X, self.Y = np.meshgrid(xs, ys)

    # -----------------------------------------------------
    # world -> heatmap index
    # -----------------------------------------------------
    def world_to_grid(self, x, y):

        gx = (x - self.xmin) / (self.xmax - self.xmin)
        gy = (y - self.ymin) / (self.ymax - self.ymin)

        gx *= (self.resolution - 1)
        gy *= (self.resolution - 1)

        return gx, gy

    # -----------------------------------------------------
    # encode
    # -----------------------------------------------------
    def encode(self, hand_point, arm_point):
        """
        hand_point: (3,)
        arm_point: (3,)

        return:
            heatmap
        """

        heatmap = np.zeros(
            (self.resolution, self.resolution),
            dtype=np.float32
        )

        # =================================================
        # ray direction
        # =================================================
        direction = arm_point - hand_point

        # =================================================
        # 限制 z 非负
        # =================================================
        #
        # 不允许射线向下传播
        #
        # 保持 XY 方向真实
        # 仅截断 dz
        # =================================================
        # direction[2] = max(direction[2], 0.0)

        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            return heatmap

        direction = direction / norm

        # =================================================
        # ray 起点
        # =================================================
        x0 = hand_point[0]
        y0 = hand_point[1]
        z0 = hand_point[2]

        # ray direction
        dx = direction[0]
        dy = direction[1]
        dz = direction[2]

        # =================================================
        # 计算 ray 与 heatmap 边界的交点
        # =================================================
        #
        # 用于限制 forward ray 的有效长度
        #
        # 同时在边缘额外延伸 2*width
        # 避免 gaussian 被硬截断
        # =================================================

        extended_xmin = self.xmin - 2 * self.width
        extended_xmax = self.xmax + 2 * self.width

        extended_ymin = self.ymin - 2 * self.width
        extended_ymax = self.ymax + 2 * self.width

        tx_candidates = []
        ty_candidates = []

        if abs(dx) > 1e-8:
            tx1 = (extended_xmin - x0) / dx
            tx2 = (extended_xmax - x0) / dx
            tx_candidates.extend([tx1, tx2])

        if abs(dy) > 1e-8:
            ty1 = (extended_ymin - y0) / dy
            ty2 = (extended_ymax - y0) / dy
            ty_candidates.extend([ty1, ty2])

        candidates = tx_candidates + ty_candidates

        # =================================================
        # 只保留 forward direction
        # =================================================
        candidates = [v for v in candidates if v > 0]

        if len(candidates) == 0:
            return heatmap

        ray_length = min(candidates)

        # =================================================
        # 用户可选最大长度限制
        # =================================================
        if self.max_ray_length is not None:
            ray_length = min(ray_length, self.max_ray_length)

        # =================================================
        # XY projection direction norm
        # =================================================
        #
        # 只在 XY 平面做投影
        # =================================================
        dxy2 = dx * dx + dy * dy

        if dxy2 < 1e-10:
            return heatmap

        # =================================================
        # heatmap 上每个像素
        # 投影到 ray 的 XY projection
        # =================================================
        #
        # ray:
        #
        #     R(t)=P0+t*d
        #
        # projection:
        #
        #     t = dot(P-P0,d)/|d|²
        #
        # =================================================

        PX = self.X - x0
        PY = self.Y - y0

        t = (PX * dx + PY * dy) / dxy2

        # =================================================
        # 只保留 forward ray
        # =================================================
        #
        # 不允许：
        #
        #     t < 0
        #
        # 同时限制在 ray_length 内
        # =================================================
        t = np.clip(t, 0.0, ray_length)

        # =================================================
        # ray 上对应最近点
        # =================================================
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy

        # =================================================
        # 到 ray 的横向距离
        # =================================================
        dist = np.sqrt(
            (self.X - proj_x) ** 2 +
            (self.Y - proj_y) ** 2
        )

        # =================================================
        # Gaussian smooth falloff
        # =================================================
        w = gaussian_weight(dist, self.width)

        # =================================================
        # projection 对应 z
        # =================================================
        #
        # z(t)=z0+t*dz
        # =================================================
        z_field = z0 + t * dz

        # =================================================
        # final heatmap
        # =================================================
        heatmap = w * z_field

        return heatmap

    # =========================================================
    # 从 heatmap 反推射线（简单版本）
    # =========================================================

def decode_heatmap_direction(heatmap):
    """
    一个最小可逆验证：

    利用 PCA 找到主方向。

    这里只是证明这个 representation
    保留了 arm 的主方向信息。
    """

    ys, xs = np.where(heatmap > np.percentile(heatmap, 90))

    if len(xs) < 2:
        return None

    pts = np.stack([xs, ys], axis=1)

    mean = pts.mean(axis=0)
    pts = pts - mean

    cov = pts.T @ pts

    eigvals, eigvecs = np.linalg.eig(cov)

    direction = eigvecs[:, np.argmax(eigvals)]

    return direction


# =========================================================
# 缺失值插值
# =========================================================

def interpolate_tracking(df):
    """
    对 hand / arm 的缺失值做时间插值。

    使用：
        pandas.interpolate

    特点：
    - 前后值平滑补全
    - 不会直接丢弃 frame
    - 时间连续性更好
    """

    cols = [
        "HAND_x", "HAND_y", "HAND_z",
        "ARM_x", "ARM_y", "ARM_z",
    ]

    # 转 float
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 双向插值
    df[cols] = df[cols].interpolate(
        method="linear",
        limit_direction="both"
    )

    return df

# =========================================================
# 时序平滑
# =========================================================

def smooth_tracking(points, alpha=0.2):
    """
    points:
        (N,3)

    alpha:
        越小越平滑
        越大越跟随实时

    推荐:
        0.15 ~ 0.3
    """

    smoothed = np.zeros_like(points)

    smoothed[0] = points[0]

    for i in range(1, len(points)):
        smoothed[i] = (
            alpha * points[i]
            + (1 - alpha) * smoothed[i - 1]
        )

    return smoothed


# =========================================================
# 读取 csv
# =========================================================

def load_csv(csv_path, plane_path="plane_calibration_flipped.npz"):

    df = pd.read_csv(csv_path)

    # =====================================================
    # 先做缺失值插值
    # =====================================================
    df = interpolate_tracking(df)

    R, t = load_plane_calibration(plane_path)

    hand_points = []
    arm_points = []

    for _, row in df.iterrows():

        hand_cam = np.array([
            row["HAND_x"],
            row["HAND_y"],
            row["HAND_z"],
        ])

        arm_cam = np.array([
            row["ARM_x"],
            row["ARM_y"],
            row["ARM_z"],
        ])

        # =============================================
        # 梯形校正（camera -> plane）
        # =============================================
        hand_plane = camera_to_plane(hand_cam, R, t)
        arm_plane = camera_to_plane(arm_cam, R, t)

        hand_points.append(hand_plane)
        arm_points.append(arm_plane)

    hand_points = np.array(hand_points)
    arm_points = np.array(arm_points)

    hand_points = smooth_tracking(
        hand_points,
        alpha=0.5
    )

    arm_points = smooth_tracking(
        arm_points,
        alpha=0.5
    )

    return hand_points, arm_points


# =========================================================
# 根据整个 csv 自动计算范围
# =========================================================

def compute_ranges(hand_points, arm_points, margin=0.05):
    """
    视野范围只由 hand position 决定。

    原因：
    arm ray 是无限延伸的。

    如果使用 arm point:
        会导致 bounding box 被方向拉爆。

    因此：
    使用 hand trajectory 定义真正的 observable workspace。
    """

    pts = hand_points

    xmin = pts[:, 0].min() - margin
    xmax = pts[:, 0].max() + margin

    ymin = pts[:, 1].min() - margin
    ymax = pts[:, 1].max() + margin

    return (xmin, xmax), (ymin, ymax)

def compute_fixed_ranges_from_training_set(
    pattern="*20260511_train*.csv",
    plane_path="plane_calibration_flipped.npz",
    margin=0.05,
):
    """
    扫描所有 training csv，
    计算统一 fixed workspace。

    bounding box 仅使用 hand trajectory。
    """

    csv_files = sorted(glob.glob(pattern))

    if len(csv_files) == 0:
        raise RuntimeError(
            f"No csv files found matching pattern: {pattern}"
        )

    all_hand_points = []

    print("\nComputing global workspace from training set:")
    for path in csv_files:

        print("  ", path)

        hand_points, _ = load_csv(
            path,
            plane_path=plane_path
        )

        all_hand_points.append(hand_points)

    all_hand_points = np.concatenate(all_hand_points, axis=0)

    xmin = all_hand_points[:, 0].min() - margin
    xmax = all_hand_points[:, 0].max() + margin

    ymin = all_hand_points[:, 1].min() - margin
    ymax = all_hand_points[:, 1].max() + margin

    print("\nFixed workspace:")
    print(f"X: [{xmin:.3f}, {xmax:.3f}]")
    print(f"Y: [{ymin:.3f}, {ymax:.3f}]")

    return (xmin, xmax), (ymin, ymax)

# =========================================================
# Visualization（手动逐帧）
# =========================================================

def visualize_heatmap_sequence(
    csv_path,
    plane_path="plane_calibration.npz",
    resolution=128,
    width=0.03,
):

    hand_points, arm_points = load_csv(csv_path, plane_path)

    x_range, y_range = compute_fixed_ranges_from_training_set(
        pattern="*20260511_train*.csv",
        plane_path=plane_path,
        margin=0.05,
    )

    encoder = ArmHeatmapEncoder(
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        width=width,
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    # =====================================================
    # 当前 frame
    # =====================================================
    current_frame = [0]

    # =====================================================
    # hand trajectory trace
    # =====================================================
    trace_x = hand_points[:, 0]
    trace_y = hand_points[:, 1]

    first_heatmap = encoder.encode(
        hand_points[0],
        arm_points[0],
    )

    im = ax.imshow(
        first_heatmap,
        origin="lower",
        cmap="jet",
        vmin=-0.2,
        vmax=0.2,
        extent=[
            x_range[0],
            x_range[1],
            y_range[0],
            y_range[1],
        ]
    )

    plt.colorbar(im, ax=ax, label="Z height")

    # =====================================================
    # hand trajectory
    # =====================================================
    ax.plot(
        trace_x,
        trace_y,
        color='white',
        alpha=0.35,
        linewidth=1.5,
        label='Hand Trace'
    )

    # =====================================================
    # true hand position
    # =====================================================
    hand_scatter = ax.scatter(
        [hand_points[0][0]],
        [hand_points[0][1]],
        c='red',
        s=80,
        marker='o',
        label='True Hand'
    )

    # =====================================================
    # inferred hand position from heatmap
    # =====================================================
    inferred_scatter = ax.scatter(
        [hand_points[0][0]],
        [hand_points[0][1]],
        c='cyan',
        s=80,
        marker='x',
        label='Inferred Hand'
    )

    ax.legend(loc='upper right')

    ax.set_title("Arm Heatmap Representation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # =====================================================
    # 更新显示
    # =====================================================
    def redraw():

        idx = current_frame[0]

        heatmap = encoder.encode(
            hand_points[idx],
            arm_points[idx],
        )

        im.set_data(heatmap)

        # =================================================
        # 更新真实 hand position
        # =================================================
        hand_xy = hand_points[idx][:2]

        hand_scatter.set_offsets([
            [hand_xy[0], hand_xy[1]]
        ])

        # =================================================
        # 从 heatmap 反推 hand 的 x,y
        # =================================================
        max_idx = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape
        )

        hy, hx = max_idx

        inferred_x = np.interp(
            hx,
            [0, resolution - 1],
            [x_range[0], x_range[1]]
        )

        inferred_y = np.interp(
            hy,
            [0, resolution - 1],
            [y_range[0], y_range[1]]
        )

        inferred_scatter.set_offsets([
            [inferred_x, inferred_y]
        ])

        ax.set_title(f"Frame {idx} / {len(hand_points)-1}")

        fig.canvas.draw_idle()

    # =====================================================
    # keyboard control
    # =====================================================
    def on_key(event):

        # 空格：前进一帧
        if event.key == ' ':

            current_frame[0] += 1

            if current_frame[0] >= len(hand_points):
                current_frame[0] = len(hand_points) - 1

            redraw()

        # backspace：后退一帧
        elif event.key == 'backspace':

            current_frame[0] -= 1

            if current_frame[0] < 0:
                current_frame[0] = 0

            redraw()

        # r：回到第一帧
        elif event.key == 'r':

            current_frame[0] = 0
            redraw()

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("Controls:")
    print("  SPACE      -> next frame")
    print("  BACKSPACE  -> previous frame")
    print("  R          -> reset")

    plt.show()


# =========================================================
# main
# =========================================================

if __name__ == "__main__":

    visualize_heatmap_sequence(
        csv_path="aruco_rfid_20260511_test_Z.csv",
        plane_path="plane_calibration_flipped.npz",
        resolution=128,
        width=0.1,
    )