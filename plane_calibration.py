import cv2
import numpy as np
import time

# ==========================
# 相机参数
# ==========================
calib = np.load("camera_calibration.npz")
K = calib["K"]
dist = calib["dist"]

# ==========================
# ArUco
# ==========================
MARKER_SIZE = 0.05
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# ==========================
# 摄像头
# ==========================
cap = cv2.VideoCapture(0)

points = []

print("Press p to collect point (ID=0)")
print("Press c to compute plane")
print("ESC to exit")

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.undistort(frame, K, dist)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, K, dist
        )

        for i, marker_id in enumerate(ids.flatten()):

            if marker_id == 0:
                tvec = tvecs[i][0]

                cv2.drawFrameAxes(frame, K, dist, rvecs[i], tvecs[i], 0.03)

                cv2.putText(frame,
                            f"Point: {tvec.round(3)}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

    cv2.imshow("Collect Plane Points", frame)

    key = cv2.waitKey(1) & 0xFF

    # ==========================
    # 采集点
    # ==========================
    if key == ord('p'):
        if ids is not None and 0 in ids:
            idx = np.where(ids.flatten() == 0)[0][0]
            tvec = tvecs[idx][0]
            points.append(tvec)
            print(f"Collected: {tvec}")

    # ==========================
    # 计算平面
    # ==========================
    elif key == ord('c'):

        pts = np.array(points)

        if len(pts) < 3:
            print("Need at least 3 points")
            continue

        # ========= 平面拟合 =========
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid

        _, _, vh = np.linalg.svd(pts_centered)
        normal = vh[-1]   # 平面法向

        # ========= 构造坐标系 =========
        z_axis = normal / np.linalg.norm(normal)

        # x轴：用第一个点方向
        x_axis = pts[1] - pts[0]
        x_axis = x_axis / np.linalg.norm(x_axis)

        # y轴：正交
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # 再正交一次保证稳定
        x_axis = np.cross(y_axis, z_axis)

        R = np.vstack([x_axis, y_axis, z_axis]).T
        t = centroid

        # 保存
        np.savez("plane_calibration.npz", R=R, t=t)

        print("✅ Plane calibration saved!")
        print("R:\n", R)
        print("t:\n", t)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()