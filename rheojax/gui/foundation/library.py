from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetRef:
    id: str
    name: str
    protocol_type: str  # a real protocol name, or "" for a derived dataset
    # whose transform output has no determinable
    # rheological protocol (design §7: "stored but
    # not offered to typed Fit slots" -- "" never
    # equality-matches a real protocol in
    # datasets_of_type() below)
    origin: str  # "imported" | "derived"
    units: dict[str, str]
    row_count: int
    hash: str
    provenance: dict[str, Any]
    lineage: list[str] = field(default_factory=list)


class DatasetLibrary:
    def __init__(self) -> None:
        self._by_id: dict[str, DatasetRef] = {}
        self._payloads: dict[
            str, Any
        ] = {}  # id -> RheoData; needed to resolve data_ref strings
        # Pipeline batch runs mutate this library from a QThreadPool worker thread
        # while the GUI thread reads it concurrently (library rail, fit/transform
        # controllers). Without this lock, add()'s pop-then-set is not atomic
        # (KeyError on a torn get()/load_payload() read) and all()/datasets_of_type()
        # can raise "dictionary changed size during iteration" mid-add()/remove().
        self._lock = threading.RLock()
        # Exposed publicly (same RLock instance) so a caller can hold it across a
        # multi-call sequence -- e.g. add() immediately followed by store_payload()
        # for the same id -- so a concurrent get()/load_payload() never observes a
        # DatasetRef whose payload hasn't been written yet. RLock: safe to reacquire
        # from within a `with library.lock:` block since add()/store_payload()
        # themselves also acquire it.
        self.lock = self._lock

    def add(self, ref: DatasetRef, overwrite: bool = False) -> None:
        with self._lock:
            if not overwrite and ref.id in self._by_id:
                raise ValueError(
                    f"Dataset id {ref.id!r} already exists (pass overwrite=True to replace)"
                )
            # An overwrite replaces the reference AND clears any stale payload under the old
            # reference -- without this, a caller that overwrites a ref but doesn't also call
            # store_payload() (e.g. a fit export whose result has no derived RheoData) would leave
            # the PREVIOUS ref's payload silently reachable under the new ref's id/metadata.
            self._payloads.pop(ref.id, None)
            self._by_id[ref.id] = ref

    def get(self, id: str) -> DatasetRef:
        with self._lock:
            return self._by_id[id]

    def remove(self, id: str) -> None:
        with self._lock:
            self._by_id.pop(id, None)
            self._payloads.pop(id, None)

    def all(self) -> list[DatasetRef]:
        with self._lock:
            return list(self._by_id.values())

    def datasets_of_type(self, protocol_type: str) -> list[DatasetRef]:
        with self._lock:
            return [
                r for r in self._by_id.values() if r.protocol_type == protocol_type
            ]

    def store_payload(self, id: str, data: Any) -> None:
        with self._lock:
            self._payloads[id] = data

    def load_payload(self, id: str) -> Any:
        with self._lock:
            return self._payloads[id]
