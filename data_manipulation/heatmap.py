import numpy as np

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
    Smooth arm field representation

    heatmap value:
        z-height weighted by distance to arm ray
    """

    H, W = heatmap_size

    heatmap = np.zeros((H, W), dtype=np.float32)

    xmin, xmax = grid_x
    ymin, ymax = grid_y

    # =====================================================
    # arm segment
    # =====================================================
    p0 = wrist_point
    p1 = wrist_point + arm_direction * arm_length

    # =====================================================
    # build XY grid
    # =====================================================
    xs = np.linspace(xmin, xmax, W)
    ys = np.linspace(ymin, ymax, H)

    X, Y = np.meshgrid(xs, ys)

    # plane points
    P = np.stack([
        X,
        Y,
        np.zeros_like(X)
    ], axis=-1)

    # =====================================================
    # segment direction
    # =====================================================
    seg = p1 - p0
    seg_len2 = np.dot(seg, seg)

    # vector from p0 to every pixel
    v = P - p0

    # projection coefficient
    t = np.sum(v * seg, axis=-1) / seg_len2

    # clamp to segment
    t = np.clip(t, 0.0, 1.0)

    # closest points on arm
    closest = p0 + t[..., None] * seg

    # =====================================================
    # distance to arm
    # =====================================================
    dist = np.linalg.norm(P - closest, axis=-1)

    # =====================================================
    # Gaussian tube
    # =====================================================
    G = np.exp(-(dist ** 2) / (2 * arm_width ** 2))

    # =====================================================
    # interpolate z along arm
    # =====================================================
    z = closest[..., 2]

    heatmap = z * G

    return heatmap