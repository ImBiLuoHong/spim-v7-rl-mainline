from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class InteractionConfig:
    """
    Configuration for interaction/intervention mechanisms.
    (Recreated stub based on usage in ModelConfig)
    """
    enabled: bool = False
    mode: str = 'none'
    strength: float = 0.0
