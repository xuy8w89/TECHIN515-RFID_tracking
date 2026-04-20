class Tag:
    def __init__(self, raw: bytes):
        """
        raw: bytes
        = FreqAnt + PC(2) + EPC(N) + RSSI(1) + Phase(2)
        """
        self.raw = raw

        idx = 0

        # —— FreqAnt ——
        self.freq_ant = raw[idx]
        idx += 1

        self.freq = (self.freq_ant >> 2) & 0x3F
        self.ant = self.freq_ant & 0x03

        # —— PC ——
        self.pc = int.from_bytes(raw[idx:idx+2], "big")
        idx += 2

        # EPC 长度（Gen2）
        self.epc_len = ((self.pc >> 11) & 0x1F) * 2

        # —— EPC ——
        self.epc = raw[idx:idx + self.epc_len]
        idx += self.epc_len

        # —— RSSI ——
        rssi_byte = raw[idx]
        rssi_byte = rssi_byte & 0x7F      # 去掉天线高位标志
        rssi_byte -= 128
        idx += 1

        # 高位用于天线扩展，不计入 RSSI
        self.rssi = rssi_byte

        # —— Phase ——
        self.phase = int.from_bytes(raw[idx:idx+2], "big")

    def filter(self, prefix: bytes, mask: bytes) -> bool:
        epc = self.epc

        if len(prefix) != len(mask):
            raise ValueError("prefix and mask must have the same length")

        if len(epc) < len(prefix):
            return False

        for i in range(len(prefix)):
            if (epc[i] & mask[i]) != (prefix[i] & mask[i]):
                return False

        return True

    def __str__(self):
        return (
            f"Tag(\n"
            f"  EPC      = {self.epc.hex().upper()},\n"
            f"  Antenna  = {self.ant + 1},\n"
            f"  FreqIdx  = {self.freq},\n"
            f"  RSSI     = {self.rssi},\n"
            f"  Phase    = {self.phase}\n"
            f"  raw    = {self.raw}\n"
            f")"
        )

    __repr__ = __str__
