import cv2
import numpy as np
from lib.reader import E710Reader
import time
import csv
from datetime import datetime

def draw_grid(frame, grid_size=50, color=(200, 200, 200), thickness=1):
    h, w = frame.shape[:2]

    # 画竖线
    for x in range(0, w, grid_size):
        cv2.line(frame, (x, 0), (x, h), color, thickness)

    # 画横线
    for y in range(0, h, grid_size):
        cv2.line(frame, (0, y), (w, y), color, thickness)

    return frame

# ==========================
# 📌 加载相机标定参数
# ==========================
calib = np.load("camera_calibration.npz")

camera_matrix = calib["K"]
dist_coeffs = calib["dist"]

print("✅ Loaded calibration")
print("K:\n", camera_matrix)
print("dist:\n", dist_coeffs)

# ==========================
# RFID 配置
# ==========================
PORT = "COM14"
reader = E710Reader(PORT)

TARGET_BYTES = bytes([0x01, 0x02, 0x03, 0x04, 0x05])

# ==========================
# ArUco 配置
# ==========================
MARKER_SIZE = 0.05  # 5cm

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# ==========================
# 录制控制
# ==========================
recording = False
record_data = []

# ==========================
# 摄像头
# ==========================
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
timestamp = None

print("Press x to start recording")
print("Press c to stop recording")
print("Press ESC to exit")

# ==========================
# 主循环
# ==========================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break
    
    timestamp = time.time()

    # ==========================
    # 📌 可选：去畸变（强烈建议开启）
    # ==========================
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # ==========================
    # 1️⃣ ArUco 检测
    # ==========================
    interaction_id = -1   # 默认无交互
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    wrist_xyz = None

    if ids is not None:

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            MARKER_SIZE,
            camera_matrix,
            dist_coeffs
        )

        wrist_xyz = None
        wrist_rvec = None   # ✅ 新增
        for i, marker_id in enumerate(ids.flatten()):

            # ======================
            # Wrist (ID = 1)
            # ======================
            if marker_id == 1:

                tvec = tvecs[i][0]
                rvec = rvecs[i][0]

                wrist_xyz = tvec
                wrist_rvec = rvec

                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    rvecs[i],
                    tvec,
                    0.03
                )

                cv2.putText(frame,
                            f"Wrist: {tvec.round(3)}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

            # ======================
            # Interaction (ID = 2~4)
            # ======================
            elif marker_id in [2, 3, 4]:

                # 只记录第一个，保证唯一性
                if interaction_id == -1:
                    interaction_id = int(marker_id)

                    cv2.putText(frame,
                                f"Interaction ID: {interaction_id}",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2)

    # ==========================
    # 2️⃣ RFID
    # ==========================
    inv = reader.fast_inventory(antennas=(0,1,2,3))
    raw_tags = inv.get("tags", [])

    rfid_data = {}

    for raw_tag in raw_tags:
        data = raw_tag["data"]

        freq_ant = data[0]
        pc = data[1:3]

        epc_len = ((pc[0] & 0xF8) >> 3) * 2
        epc_bytes = data[3:3+epc_len]

        if len(epc_bytes) < 7 or epc_bytes[2:7] != TARGET_BYTES:
            continue

        epc_hex = epc_bytes.hex()

        rssi_byte = data[3+epc_len]
        phase_bytes = data[4+epc_len:6+epc_len]
        phase = int.from_bytes(phase_bytes, byteorder="big")

        antenna_id = (freq_ant & 0x03) + 1
        rssi = rssi_byte & 0x7F

        key = (antenna_id, epc_hex)
        rfid_data[key] = (rssi, phase)

    # ==========================
    # 3️⃣ 记录
    # ==========================
    if recording:

        row = {
            "timestamp": timestamp,
            "interaction_id": interaction_id
        }

        if wrist_xyz is not None:
            row["WRIST_x"] = wrist_xyz[0]
            row["WRIST_y"] = wrist_xyz[1]
            row["WRIST_z"] = wrist_xyz[2]

        if wrist_rvec is not None:
            row["WRIST_rx"] = wrist_rvec[0]
            row["WRIST_ry"] = wrist_rvec[1]
            row["WRIST_rz"] = wrist_rvec[2]

        for (antenna_id, epc), (rssi, phase) in rfid_data.items():
            row[f"ant{antenna_id}_{epc}_rssi"] = rssi
            row[f"ant{antenna_id}_{epc}_phase"] = phase

        record_data.append(row)

    # ==========================
    # 显示
    # ==========================
    tag_count = len(rfid_data)

    cv2.putText(frame,
                f"Tags: {tag_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)
    frame = draw_grid(frame, grid_size=60)

    cv2.imshow("ArUco Wrist + RFID", frame)

   

    key = cv2.waitKey(1) & 0xFF

    if key == ord('x'):
        record_data = []
        recording = True
        print("▶ Recording started")

    elif key == ord('c'):
        if recording:
            recording = False

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aruco_rfid_{ts}.csv"

            if len(record_data) > 0:

                all_keys = set()
                for row in record_data:
                    all_keys.update(row.keys())

                all_keys = sorted(all_keys)

                with open(filename, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=all_keys)
                    writer.writeheader()
                    for row in record_data:
                        writer.writerow(row)

                print(f"■ Saved to {filename}")

    elif key == 27:
        break

reader.close()
cap.release()
cv2.destroyAllWindows()