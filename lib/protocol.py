# e710/protocol.py

def e710_checksum(data: bytes) -> int:
    return ((~(sum(data) & 0xFF)) + 1) & 0xFF


def build_cmd(cmd: int, data: bytes = b"", addr: int = 0xFF) -> bytes:
    length = 1 + 1 + len(data) + 1  # addr + cmd + data + check

    frame_wo_check = bytes([
        0xA0,
        length,
        addr,
        cmd
    ]) + data

    check = e710_checksum(frame_wo_check)
    return frame_wo_check + bytes([check])


def receive_frame(read_byte_func):
    # 找帧头
    while True:
        b = read_byte_func()
        if not b:
            raise TimeoutError("No data")
        if b[0] == 0xA0:
            break

    length = read_byte_func()[0]

    payload = b""
    while len(payload) < length:
        chunk = read_byte_func()
        if not chunk:
            raise TimeoutError("Frame incomplete")
        payload += chunk

    frame = bytes([0xA0, length]) + payload

    recv_check = frame[-1]
    calc_check = e710_checksum(frame[:-1])
    if recv_check != calc_check:
        raise ValueError("Checksum error")

    return {
        "addr": payload[0],
        "cmd": payload[1],
        "data": payload[2:-1],
        "raw": frame
    }
