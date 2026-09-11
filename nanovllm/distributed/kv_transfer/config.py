"""Configuration contracts for the first, single-host PD implementation.

Constructing the config does not start a transport; ModelRunner owns startup.
"""
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from uuid import uuid4


class PDRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class PDConfig:
    role: PDRole
    # Stable for the lifetime of one engine, unique across engine restarts.
    engine_id: str = field(default_factory=lambda: uuid4().hex)
    connector: str = "nixl"
    host: str = "127.0.0.1"
    port: int = 5600
    transfer_timeout: float = 60.0

    def __post_init__(self):
        object.__setattr__(self, "role", PDRole(self.role))
        if not self.engine_id.strip() or not self.host.strip():
            raise ValueError("PD engine_id and host must be nonempty")
        if self.connector != "nixl":
            raise ValueError("The initial PD connector is nixl")
        if not 1 <= self.port <= 65535:
            raise ValueError("PD port must be between 1 and 65535")
        if not isfinite(self.transfer_timeout) or self.transfer_timeout <= 0:
            raise ValueError("PD transfer_timeout must be finite and positive")
