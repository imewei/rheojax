from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetRef:
    id: str
    name: str
    protocol_type: str               # a real protocol name, or "" for a derived dataset
                                      # whose transform output has no determinable
                                      # rheological protocol (design §7: "stored but
                                      # not offered to typed Fit slots" -- "" never
                                      # equality-matches a real protocol in
                                      # datasets_of_type() below)
    origin: str                      # "imported" | "derived"
    units: dict[str, str]
    row_count: int
    hash: str
    provenance: dict[str, Any]
    lineage: list[str] = field(default_factory=list)

class DatasetLibrary:
    def __init__(self) -> None:
        self._by_id: dict[str, DatasetRef] = {}
        self._payloads: dict[str, Any] = {}  # id -> RheoData; needed to resolve data_ref strings

    def add(self, ref: DatasetRef) -> None:
        self._by_id[ref.id] = ref

    def get(self, id: str) -> DatasetRef:
        return self._by_id[id]

    def remove(self, id: str) -> None:
        self._by_id.pop(id, None)
        self._payloads.pop(id, None)

    def all(self) -> list[DatasetRef]:
        return list(self._by_id.values())

    def datasets_of_type(self, protocol_type: str) -> list[DatasetRef]:
        return [r for r in self._by_id.values() if r.protocol_type == protocol_type]

    def store_payload(self, id: str, data: Any) -> None:
        self._payloads[id] = data

    def load_payload(self, id: str) -> Any:
        return self._payloads[id]
