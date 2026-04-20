# e710/commands.py

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandDef:
    code: int
    name: str
    desc: str


# ===== 指令定义区 =====

CMD_GET_FREQUENCY_REGION = CommandDef(
    code=0x79,
    name="GET_FREQUENCY_REGION",
    desc="Query RF frequency region"
)

CMD_FAST_INVENTORY = CommandDef(
    code=0x8A,
    name="FAST_INVENTORY",
    desc="Fast inventory implementation 2"
)

# ===== 指令注册表 =====

COMMANDS = {
    CMD_GET_FREQUENCY_REGION.code: CMD_GET_FREQUENCY_REGION,
    CMD_FAST_INVENTORY.code: CMD_FAST_INVENTORY,
}
