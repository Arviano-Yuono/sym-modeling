from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CaseMetadata:
    """Minimal metadata shared across domain-specific case containers."""

    case_name: Optional[str] = None
    source_path: Optional[str] = None
    coordinate_system: str = "cartesian"
    tags: List[str] = field(default_factory=list)
    extras: Dict[str, object] = field(default_factory=dict)
