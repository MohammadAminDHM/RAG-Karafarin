from typing import Dict, List, Protocol, Sequence


class VectorStoreProtocol(Protocol):
    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[Dict]:
        ...
