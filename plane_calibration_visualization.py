import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ==========================
# 加载参数
# ==========================
calib = np.load("camera_calibration.npz")
K = calib["K"]
dist = calib["dist"]

plane = np.load("plane_calibration.npz")
R = plane["R"]
t = plane["t"]

# ==========================
# ArUco
# ==========================
MARKER_SIZE = 0.05
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# ==========================
# matplotlib
# ==========================
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

trajectory = deque(maxlen=100)

# ==========================
# 摄像头
# ==========================
cap = cv2.VideoCapture(0)

running = True

while cap.isOpened() and running:

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

            if marker_id == 1:

                Pc = tvecs[i][0]

                # 坐标转换
                Pp = R.T @ (Pc - t)

                # ⭐ 保存轨迹
                trajectory.append(Pp)

                # 显示
                cv2.putText(frame,
                            f"Plane: {Pp.round(3)}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

                # ========= 画3D =========
                ax.cla()

                traj = np.array(trajectory)

                # 当前点
                ax.scatter(Pp[0], Pp[1], Pp[2], c='r', s=50)

                # ⭐ 轨迹线
                if len(traj) > 1:
                    ax.plot(traj[:,0], traj[:,1], traj[:,2], c='b')

                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(-0.5, 0.5)
                ax.set_zlim(-0.5, 0.5)

                ax.set_xlabel("X (plane)")
                ax.set_ylabel("Y (plane)")
                ax.set_zlabel("Z (height)")

                plt.draw()
                plt.pause(0.001)

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    # ⭐ 按 q 或 ESC 都退出
    if key == ord('q') or key == 27:
        running = False
        break

# ==========================
# 彻底释放
# ==========================
cap.release()
cv2.destroyAllWindows()

plt.ioff()
plt.close(fig)   # ⭐ 关键：关闭 matplotlib