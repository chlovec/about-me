from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchExcept,
    MatchValue,
    PointStruct,
    Prefetch,
    QueryRequest,
    Range,
    Rrf,
    RrfQuery,
    ScoredPoint,
    SparseVectorParams,
    UpdateStatus,
    VectorParams,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)

from search_core.models import (
    EmbeddedDocument,
    EmbeddedQuery,
    SearchConfig,
    SearchMode,
    SearchResponse,
    SearchResult,
)


@dataclass
class QdrantStoreConfig:
    """Configuration parameters for establishing a Qdrant connection."""

    # Required parameters
    collection_name: str

    # Search and Storage defaults
    distance: Distance = Distance.COSINE
    vector_size: int | None = None

    # Connection routing parameters
    location: str | None = None
    url: str | None = None
    host: str | None = None
    port: int | None = None
    api_key: str | None = None

    # Escape hatch for any extra QdrantClient arguments
    client_kwargs: dict[str, Any] = field(default_factory=dict)


class QdrantEmbeddingStore:
    """A vector store implementation using Qdrant as the backend provider."""

    DOC_ID = "doc_id"
    TEXT = "text"

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        if not self.client.collection_exists(self.collection_name):
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.vector_size,
                            distance=self.distance,
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(modifier=None),
                    },
                )
            except UnexpectedResponse as e:
                # Handle race conditions in highly concurrent cluster assignments cleanly
                if "already exists" not in str(e).lower():
                    raise

    @classmethod
    def connect(cls, config: QdrantStoreConfig) -> "QdrantEmbeddingStore":
        connection_args = {
            k: v
            for k, v in {
                "location": config.location,
                "url": config.url,
                "host": config.host,
                "port": config.port,
                "api_key": config.api_key,
            }.items()
            if v is not None
        }

        client = QdrantClient(**connection_args, **config.client_kwargs)

        return cls(
            client=client,
            collection_name=config.collection_name,
            vector_size=config.vector_size,
            distance=config.distance,
        )

    # ===========================
    # Helpers
    # ===========================

    from typing import Any

    def _parse_filters(self, filters: dict[str, Any] | None) -> Filter | None:
        """Parses a dictionary into Qdrant match, exclusion, and range filters."""
        if not filters:
            return None

        must_conditions = []
        for k, v in filters.items():
            key = f"metadata.{k}"

            # 1. Handle implicit IN clauses
            if isinstance(v, (list, tuple)):
                must_conditions.append(FieldCondition(key=key, match=MatchAny(any=list(v))))

            # 2. Handle explicitly defined operators via nested dictionaries
            elif isinstance(v, dict):
                must_conditions.extend(self._parse_dict_operators(key, v))

            # 3. Handle implicit exact matches
            else:
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))

        return Filter(must=must_conditions)

    def _parse_dict_operators(self, key: str, op_dict: dict[str, Any]) -> list[FieldCondition]:
        """Helper to parse specific operator dictionaries."""
        conditions = []
        range_kwargs = {}

        # Map simple operators to their respective match/range generators
        range_ops = {"$gt": "gt", "$gte": "gte", "$lt": "lt", "$lte": "lte"}

        for op, val in op_dict.items():
            if op in range_ops:
                range_kwargs[range_ops[op]] = val
            elif op == "$ne":
                conditions.append(FieldCondition(key=key, match=MatchExcept(**{"except": [val]})))
            elif op == "$nin":
                conditions.append(
                    FieldCondition(key=key, match=MatchExcept(**{"except": list(val)}))
                )
            elif op == "$eq":
                conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
            elif op == "$in":
                conditions.append(FieldCondition(key=key, match=MatchAny(any=list(val))))

        if range_kwargs:
            conditions.append(FieldCondition(key=key, range=Range(**range_kwargs)))

        return conditions

    def _build_search_request(
        self,
        queries: Sequence[EmbeddedQuery],  # Updated to accept generic Sequence
        qfilter: Filter | None,
        payload: list[str],
        config: SearchConfig,
    ) -> list[QueryRequest]:
        requests = []
        for q in queries:
            # Fall back to dense search if requested or if sparse embeddings are missing
            if config.mode == SearchMode.DENSE or q.sparse_vector is None:
                requests.append(
                    QueryRequest(
                        query=q.embedding.tolist(),
                        using="dense",
                        filter=qfilter,
                        limit=config.k,
                        with_payload=payload,
                    )
                )
                continue

            requests.append(
                QueryRequest(
                    prefetch=[
                        Prefetch(
                            query=q.embedding.tolist(),
                            using="dense",
                            filter=qfilter,
                            limit=config.prefetch_k,
                        ),
                        Prefetch(
                            query=QdrantSparseVector(
                                indices=q.sparse_vector.indices,
                                values=q.sparse_vector.values,
                            ),
                            using="sparse",
                            filter=qfilter,
                            limit=config.prefetch_k,
                        ),
                    ],
                    query=RrfQuery(rrf=Rrf(k=config.rrf_k)),
                    limit=config.k,
                    with_payload=payload,
                )
            )
        return requests

    def _build_search_result(self, point: ScoredPoint) -> SearchResult:
        payload = point.payload or {}
        return SearchResult(
            id=payload.get(self.DOC_ID, point.id),
            text=payload.get(self.TEXT, ""),
            score=point.score,
            metadata=payload.get("metadata", {}),
        )

    # ===========================
    # Indexing
    # ===========================

    def create_metadata_index(self, field_name: str) -> None:
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=f"metadata.{field_name}",
            field_schema="keyword",
        )

    # ===========================
    # Data Ingestion
    # ===========================

    def save_embeddings(self, documents: list[EmbeddedDocument], wait: bool) -> int:
        points = []
        for doc in documents:
            doc.validate(self.vector_size)
            vector = {"dense": doc.embedding.tolist()}

            if doc.sparse_vector is not None:
                vector["sparse"] = {
                    "indices": doc.sparse_vector.indices,
                    "values": doc.sparse_vector.values,
                }

            payload = {
                self.TEXT: doc.text,
                self.DOC_ID: str(doc.id),
                "metadata": doc.metadata,
            }

            points.append(
                PointStruct(
                    id=doc.id,
                    vector=vector,
                    payload=payload,
                )
            )

        res = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=wait,
        )

        # Fail with runtime error if result status does not match expectation
        expected_status = UpdateStatus.COMPLETED if wait else UpdateStatus.ACKNOWLEDGED
        if res.status != expected_status:
            raise RuntimeError(
                f"Upsert failed with status={res.status} (expected {expected_status})"
            )

        return len(documents)

    # ===========================
    # Search
    # ===========================

    def search(
        self,
        queries: Sequence[EmbeddedQuery],
        filters: Mapping[str, Any] | None,
        config: SearchConfig,
    ) -> Iterator[SearchResponse]:
        # Create and submit the request
        responses = self.client.query_batch_points(
            collection_name=self.collection_name,
            requests=self._build_search_request(
                queries=queries,
                qfilter=self._parse_filters(filters),
                payload=(
                    [self.TEXT, self.DOC_ID]
                    + [f"metadata.{k}" for k in (config.return_metadata or [])]
                ),
                config=config,
            ),
        )

        # Process response
        if len(responses) != len(queries):
            raise RuntimeError("Qdrant returned a different number of responses than requests.")

        return iter(
            SearchResponse(
                id=query.id,
                matches=[self._build_search_result(point) for point in response.points],
            )
            for response, query in zip(responses, queries, strict=True)
        )
