# e710/reader.py

import serial
import time
from .protocol import build_cmd, receive_frame
from .commands import *


class E710Reader:
    def __init__(self, port: str, baudrate: int = 115200, addr: int = 0xFF, timeout=1):
        self.addr = addr
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def _read_byte(self):
        return self.ser.read(1)

    def _send_and_receive_all(self, cmd: int, data: bytes = b"", timeout=2.0):
        frame = build_cmd(cmd, data, self.addr)
        self.ser.write(frame)

        frames = []
        start = time.time()

        while True:
            if time.time() - start > timeout:
                raise TimeoutError("Command timeout")

            resp = receive_frame(self._read_byte)
            frames.append(resp)

            # —— 判断是否是“命令完成帧” ——
            if self._is_command_done(cmd, resp):
                break

        return frames

    def _is_command_done(self, cmd: int, frame: dict) -> bool:
        length = len(frame["raw"]) - 2  # raw = A0 + Len + payload

        # fast_switch_ant_inventory / inventory
        if cmd in (0x80, 0x8A, 0x89, 0x8B):
            # 结束帧长度是固定的
            return length in (0x0A, 0x0C)

        # 默认：单应答命令
        return True
    
    def _send_and_recv(self, cmd: int, data: bytes = b""):
        frame = build_cmd(cmd, data, self.addr)
        self.ser.write(frame)
        return receive_frame(self._read_byte)

    # ===== 高层接口 =====

    def get_frequency_region(self):
        resp = self._send_and_recv(CMD_GET_FREQUENCY_REGION.code)

        data = resp["data"]
        return {
            "region": data[0],
            "start_freq": data[1],
            "end_freq": data[2],
            "raw": resp
        }

    def fast_inventory(
        self,
        antennas=(0, 3),   # 👈 默认只开 0 和 3
        stay=1,
        interval=0,
        repeat=1,
        session=0,   # S0
        target=0,    # A
        phase=1      # 0=off, 1=on
    ):
        """
        CMD_FAST_SWITCH_ANT_INVENTORY (extended format, Len=0x20)
        """

        # 初始化 8 个天线（默认关闭）
        buf = []
        for i in range(8):
            if i in antennas:
                buf.extend([i, stay])   # 开启该天线
            else:
                buf.extend([0xFF, 0x00])  # 关闭

        buf += [
            interval,
            0x00, 0x00, 0x00, 0x00, 0x00,  # Reserved
            session,
            target,
            0x00, 0x00, 0x00,              # Reserved
            phase,
            repeat,
        ]

        frames = self._send_and_receive_all(0x8A, bytes(buf))

        tag_frames = []
        done_frame = None

        for f in frames:
            raw = f["raw"]
            length = raw[1]

            # Tag data frame
            if length >= 0x15:
                tag_frames.append(f)

            # Done frame
            elif length == 0x0A:
                done_frame = f

        return {
            "tags": tag_frames,
            "summary": done_frame
        }


    def close(self):
        self.ser.close()
