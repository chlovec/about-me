from unittest.mock import MagicMock, patch

import numpy as np
import pytest
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
    ScoredPoint,
    SparseVectorParams,
    UpdateResult,
    UpdateStatus,
    VectorParams,
)

from search_core.models import (
    EmbeddedDocument,
    EmbeddedQuery,
    SearchConfig,
    SearchMode,
    SearchResponse,
    SearchResult,
    SparseVector,
)
from search_core.stores import QdrantEmbeddingStore, QdrantStoreConfig

# ==============================================================================
# Pytest Fixtures
# ==============================================================================


@pytest.fixture
def default_payload():
    return ["text", "doc_id"]


@pytest.fixture
def collection_name():
    return "production_collection"


@pytest.fixture
def mock_client():
    """Provides a genuinely mocked QdrantClient that pretends the collection exists."""
    client = MagicMock()
    client.collection_exists.return_value = True
    return client


@pytest.fixture
def mock_empty_client():
    """Provides a mocked QdrantClient that pretends the collection does NOT exist."""
    client = MagicMock()
    client.collection_exists.return_value = False
    return client


@pytest.fixture
def store(mock_client, collection_name):
    """Provides a standard QdrantEmbeddingStore instance for behavioral tests."""
    return QdrantEmbeddingStore(client=mock_client, collection_name=collection_name, vector_size=3)


@pytest.fixture
def sample_queries():
    """Provides a base realistic query sequence used by calling systems."""
    return [
        EmbeddedQuery(
            id="query_id_1",
            text="machine learning",
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sparse_vector=None,
        )
    ]


# ==============================================================================
# 1. Connection & Initialization Tests
# ==============================================================================


class TestQdrantEmbeddingStoreConnect:
    @patch("search_core.stores.qdrant_embedding_store.QdrantClient")
    @patch(
        "search_core.stores.qdrant_embedding_store.QdrantEmbeddingStore.__init__", return_value=None
    )
    def test_connect_filters_none_and_unpacks_kwargs(self, mock_init, mock_qdrant_client_cls):
        """Verifies that connect filters out None parameters, merges client_kwargs, and invokes correctly."""
        config = QdrantStoreConfig(
            collection_name="production_collection",
            vector_size=512,
            distance=Distance.DOT,
            url="https://localhost:6333",
            api_key="super-secret-key",
            location=None,  # Should be filtered out
            host=None,  # Should be filtered out
            port=None,  # Should be filtered out
            client_kwargs={"timeout": 60, "grpc_port": 6334},
        )

        store = QdrantEmbeddingStore.connect(config)

        mock_qdrant_client_cls.assert_called_once_with(
            url="https://localhost:6333",
            api_key="super-secret-key",
            timeout=60,
            grpc_port=6334,
        )

        mock_init.assert_called_once_with(
            client=mock_qdrant_client_cls.return_value,
            collection_name="production_collection",
            vector_size=512,
            distance=Distance.DOT,
        )

        assert isinstance(store, QdrantEmbeddingStore)

    @patch("search_core.stores.qdrant_embedding_store.QdrantClient")
    @patch(
        "search_core.stores.qdrant_embedding_store.QdrantEmbeddingStore.__init__", return_value=None
    )
    def test_connect_handles_minimal_config(self, mock_init, mock_qdrant_client_cls):
        """Verifies connect behavior when only mandatory config properties are provided."""
        config = QdrantStoreConfig(collection_name="minimal_collection", vector_size=128)

        QdrantEmbeddingStore.connect(config)

        mock_qdrant_client_cls.assert_called_once_with()

        mock_init.assert_called_once_with(
            client=mock_qdrant_client_cls.return_value,
            collection_name="minimal_collection",
            vector_size=128,
            distance=Distance.COSINE,  # Default
        )


class TestQdrantEmbeddingStoreInit:
    def test_init_creates_collection_if_not_exists(self, mock_empty_client):
        QdrantEmbeddingStore(
            client=mock_empty_client,
            collection_name="test_collection",
            vector_size=128,
            distance=Distance.COSINE,
        )

        mock_empty_client.collection_exists.assert_called_once_with("test_collection")
        mock_empty_client.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vectors_config={"dense": VectorParams(size=128, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(modifier=None)},
        )

    def test_init_skips_creation_if_collection_exists(self, mock_client):
        QdrantEmbeddingStore(
            client=mock_client,
            collection_name="test_collection",
            vector_size=128,
        )
        mock_client.create_collection.assert_not_called()

    def test_init_raises_unexpected_response_if_generic_error(self, mock_empty_client):
        mock_empty_client.create_collection.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Error",
            content=b"Internal Server Error Context",
            headers={"X-Test-Header": "error"},
        )

        with pytest.raises(UnexpectedResponse) as exc_info:
            QdrantEmbeddingStore(
                client=mock_empty_client,
                collection_name="test_collection",
                vector_size=128,
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.reason_phrase == "Internal Error"
        assert exc_info.value.headers == {"X-Test-Header": "error"}
        assert exc_info.value.content == b"Internal Server Error Context"

    def test_init_handles_race_condition_cleanly_when_already_exists(self, mock_empty_client):
        mock_exception = UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=b"Collection test_collection already exists!",
            headers={},
        )
        mock_empty_client.create_collection.side_effect = mock_exception

        store = QdrantEmbeddingStore(
            client=mock_empty_client,
            collection_name="test_collection",
            vector_size=128,
        )

        assert store.collection_name == "test_collection"


# ==============================================================================
# 2. Write & Indexing Operations
# ==============================================================================


class TestCreateMetadataIndexes:
    def test_create_metadata_indexes_success(self, store, collection_name):
        store.create_metadata_index("user_name")

        store.client.create_payload_index.assert_called_once_with(
            collection_name=collection_name,
            field_name="metadata.user_name",
            field_schema="keyword",
        )


class TestSaveEmbeddings:
    def test_save_embeddings_dense_only_wait_true(self, store, collection_name):
        mock_doc = MagicMock(spec=EmbeddedDocument)
        mock_doc.id = 101
        mock_doc.text = "Sample text payload"
        mock_doc.embedding = np.array([0.1, 0.2, 0.3])
        mock_doc.sparse_vector = None
        mock_doc.metadata = {"category": "test"}

        store.client.upsert.return_value = UpdateResult(
            operation_id=1, status=UpdateStatus.COMPLETED
        )

        result_count = store.save_embeddings([mock_doc], wait=True)

        assert result_count == 1
        mock_doc.validate.assert_called_once_with(store.vector_size)

        store.client.upsert.assert_called_once_with(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=101,
                    vector={"dense": [0.1, 0.2, 0.3]},
                    payload={
                        "text": "Sample text payload",
                        "doc_id": "101",
                        "metadata": {"category": "test"},
                    },
                )
            ],
        )

    def test_save_embeddings_hybrid_wait_false(self, store, collection_name):
        mock_doc = MagicMock(spec=EmbeddedDocument)
        mock_doc.id = 202
        mock_doc.text = "Hybrid vector text"
        mock_doc.embedding = np.array([0.9, 0.8, 0.7])

        mock_sparse = MagicMock()
        mock_sparse.indices = [12, 45]
        mock_sparse.values = [0.35, 0.88]
        mock_doc.sparse_vector = mock_sparse
        mock_doc.metadata = {}

        store.client.upsert.return_value = UpdateResult(
            operation_id=2, status=UpdateStatus.ACKNOWLEDGED
        )

        result_count = store.save_embeddings([mock_doc], wait=False)

        assert result_count == 1
        mock_doc.validate.assert_called_once_with(store.vector_size)

        store.client.upsert.assert_called_once_with(
            collection_name=collection_name,
            wait=False,
            points=[
                PointStruct(
                    id=202,
                    vector={
                        "dense": [0.9, 0.8, 0.7],
                        "sparse": {"indices": [12, 45], "values": [0.35, 0.88]},
                    },
                    payload={
                        "text": "Hybrid vector text",
                        "doc_id": "202",
                        "metadata": {},
                    },
                )
            ],
        )

    @pytest.mark.parametrize(
        "wait_param, returned_status",
        [
            (True, UpdateStatus.ACKNOWLEDGED),
            (False, UpdateStatus.COMPLETED),
        ],
    )
    def test_save_embeddings_status_mismatch_raises_runtime_error(
        self, store, wait_param, returned_status
    ):
        mock_doc = MagicMock(spec=EmbeddedDocument)
        mock_doc.id = 500
        mock_doc.text = "Error vector text"
        mock_doc.embedding = np.array([0.0, 0.0, 0.0])
        mock_doc.sparse_vector = None
        mock_doc.metadata = {}

        store.client.upsert.return_value = UpdateResult(operation_id=99, status=returned_status)

        with pytest.raises(RuntimeError, match=f"Upsert failed with status={returned_status}"):
            store.save_embeddings([mock_doc], wait=wait_param)


# ==============================================================================
# 3. Search & Retrieval Tests
# ==============================================================================


class TestQdrantEmbeddingStoreSearch:
    def test_search_pipeline_with_complex_filtering(
        self, mock_client, store, sample_queries, collection_name
    ):
        mock_api_response = MagicMock()
        mock_api_response.points = [
            ScoredPoint(
                id="point-1",
                version=1,
                score=0.89,
                payload={
                    "doc_id": "doc-abc",
                    "text": "Result text",
                    "metadata": {"status": "active"},
                },
            )
        ]
        mock_client.query_batch_points.return_value = [mock_api_response]

        filters = {
            "status": "active",
            "age": {"$gte": 21, "$lte": 65},
            "tags": {"$nin": ["archived"]},
        }
        config = SearchConfig(mode=SearchMode.DENSE, k=1, return_metadata=["status"])

        results = list(store.search(sample_queries, filters=filters, config=config))

        expected_filter = Filter(
            must=[
                FieldCondition(key="metadata.status", match=MatchValue(value="active")),
                FieldCondition(key="metadata.age", range=Range(gte=21, lte=65)),
                FieldCondition(key="metadata.tags", match=MatchExcept(**{"except": ["archived"]})),
            ]
        )

        expected_request = QueryRequest(
            query=sample_queries[0].embedding.tolist(),
            filter=expected_filter,
            limit=1,
            with_payload=["text", "doc_id", "metadata.status"],
            using="dense",
        )

        mock_client.query_batch_points.assert_called_once_with(
            collection_name=collection_name, requests=[expected_request]
        )

        expected_responses = [
            SearchResponse(
                id="query_id_1",
                query="machine learning",
                matches=[
                    SearchResult(
                        id="doc-abc", score=0.89, text="Result text", metadata={"status": "active"}
                    )
                ],
            )
        ]
        assert results == expected_responses

    def test_search_pipeline_empty_filter(
        self, mock_client, store, sample_queries, collection_name, default_payload
    ):
        mock_api_response = MagicMock()
        mock_api_response.points = []
        mock_client.query_batch_points.return_value = [mock_api_response]

        config = SearchConfig(mode=SearchMode.DENSE)

        list(store.search(sample_queries, filters={}, config=config))

        expected_request = QueryRequest(
            query=sample_queries[0].embedding.tolist(),
            filter=None,
            limit=10,
            with_payload=default_payload,
            using="dense",
        )

        mock_client.query_batch_points.assert_called_once_with(
            collection_name=collection_name, requests=[expected_request]
        )

    @pytest.mark.parametrize(
        "invalid_filter",
        [
            {"nested": {"unsupported_direct_match": "value"}},
            {"another_nested": {"sub_key": "sub_value"}},
        ],
    )
    def test_search_pipeline_invalid_filter_bubble_up(self, store, sample_queries, invalid_filter):
        config = SearchConfig(mode=SearchMode.DENSE)
        with pytest.raises(ValueError, match="Direct dictionary filtering on nested object"):
            list(store.search(sample_queries, filters=invalid_filter, config=config))

    def test_search_pipeline_with_mixed_and_edge_case_filters(
        self, mock_client, store, sample_queries, collection_name, default_payload
    ):
        mock_api_response = MagicMock()
        mock_api_response.points = [
            ScoredPoint(
                id=1,
                version=1,
                score=0.99,
                payload={"doc_id": "doc-1", "text": "edge case", "metadata": {}},
            )
        ]
        mock_client.query_batch_points.return_value = [mock_api_response]

        filters = {
            "categories": ["electronics", "appliances"],
            "regions": ("US-East", "US-West"),
            "metrics": {
                "$unknown_op": 100,
                "$eq": "verified",
                "$ne": "flagged",
                "$in": [1, 2],
                "$nin": [8, 9],
                "$gt": 5,
                "$gte": 10,
                "$lt": 50,
                "$lte": 45,
            },
        }
        config = SearchConfig(mode=SearchMode.DENSE)

        results = list(store.search(sample_queries, filters=filters, config=config))

        expected_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.categories", match=MatchAny(any=["electronics", "appliances"])
                ),
                FieldCondition(key="metadata.regions", match=MatchAny(any=["US-East", "US-West"])),
                FieldCondition(key="metadata.metrics", match=MatchValue(value="verified")),
                FieldCondition(
                    key="metadata.metrics", match=MatchExcept(**{"except": ["flagged"]})
                ),
                FieldCondition(key="metadata.metrics", match=MatchAny(any=[1, 2])),
                FieldCondition(key="metadata.metrics", match=MatchExcept(**{"except": [8, 9]})),
                FieldCondition(key="metadata.metrics", range=Range(gt=5, gte=10, lt=50, lte=45)),
            ]
        )

        expected_request = QueryRequest(
            query=sample_queries[0].embedding.tolist(),
            filter=expected_filter,
            limit=10,
            with_payload=default_payload,
            using="dense",
        )

        mock_client.query_batch_points.assert_called_once_with(
            collection_name=collection_name, requests=[expected_request]
        )

        expected_responses = [
            SearchResponse(
                id="query_id_1",
                query="machine learning",
                matches=[SearchResult(id="doc-1", score=0.99, text="edge case", metadata={})],
            )
        ]
        assert results == expected_responses

    def test_search_batching_multiple_queries(
        self, mock_client, store, collection_name, default_payload
    ):
        mock_api_res1 = MagicMock()
        mock_api_res1.points = [
            ScoredPoint(
                id="doc-1",
                version=1,
                score=0.9,
                payload={"doc_id": "doc-1", "text": "t1", "metadata": {}},
            )
        ]
        mock_api_res2 = MagicMock()
        mock_api_res2.points = [
            ScoredPoint(
                id="doc-2",
                version=1,
                score=0.8,
                payload={"doc_id": "doc-2", "text": "t2", "metadata": {}},
            )
        ]

        mock_client.query_batch_points.return_value = [mock_api_res1, mock_api_res2]

        queries = [
            EmbeddedQuery(
                id="batch_q1",
                text="first",
                embedding=np.array([0.1, 0.1, 0.1], dtype=np.float32),
                sparse_vector=None,
            ),
            EmbeddedQuery(
                id="batch_q2",
                text="second",
                embedding=np.array([0.2, 0.2, 0.2], dtype=np.float32),
                sparse_vector=None,
            ),
        ]
        config = SearchConfig(mode=SearchMode.DENSE)

        results = list(store.search(queries, filters=None, config=config))

        req1 = QueryRequest(
            query=queries[0].embedding.tolist(),
            limit=10,
            with_payload=default_payload,
            using="dense",
        )
        req2 = QueryRequest(
            query=queries[1].embedding.tolist(),
            limit=10,
            with_payload=default_payload,
            using="dense",
        )

        mock_client.query_batch_points.assert_called_once_with(
            collection_name=collection_name, requests=[req1, req2]
        )

        expected_responses = [
            SearchResponse(
                id="batch_q1",
                query="first",
                matches=[SearchResult(id="doc-1", score=0.9, text="t1", metadata={})],
            ),
            SearchResponse(
                id="batch_q2",
                query="second",
                matches=[SearchResult(id="doc-2", score=0.8, text="t2", metadata={})],
            ),
        ]
        assert results == expected_responses

    def test_search_hybrid_routing_with_sparse_data(
        self, mock_client, store, collection_name, default_payload
    ):
        mock_api_response = MagicMock()
        mock_api_response.points = []
        mock_client.query_batch_points.return_value = [mock_api_response]

        hybrid_queries = [
            EmbeddedQuery(
                id="hybrid_1",
                text="hybrid match",
                embedding=np.array([0.5, 0.5, 0.5], dtype=np.float32),
                sparse_vector=SparseVector(indices=[0, 10], values=[0.3, 0.7]),
            )
        ]
        config = SearchConfig(mode=SearchMode.HYBRID, k=5, prefetch_k=20, rrf_k=60)

        results = list(store.search(hybrid_queries, filters=None, config=config))

        expected_request = QueryRequest(
            prefetch=[
                Prefetch(query=hybrid_queries[0].embedding.tolist(), using="dense", limit=20),
                Prefetch(
                    query={"indices": [0, 10], "values": [0.3, 0.7]}, using="sparse", limit=20
                ),
            ],
            query={"rrf": {"k": 60}},
            limit=5,
            with_payload=default_payload,
        )

        mock_client.query_batch_points.assert_called_once_with(
            collection_name=collection_name, requests=[expected_request]
        )

        assert results == [SearchResponse(id="hybrid_1", query="hybrid match", matches=[])]

    def test_search_hybrid_fallback_behavior_when_sparse_missing(
        self, mock_client, store, sample_queries, collection_name, default_payload
    ):
        mock_api_response = MagicMock()
        mock_api_response.points = []
        mock_client.query_batch_points.return_value = [mock_api_response]

        config = SearchConfig(mode=SearchMode.HYBRID, k=5)

        results = list(store.search(sample_queries, filters=None, config=config))

        expected_request = QueryRequest(
            query=sample_queries[0].embedding.tolist(),
            using="dense",
            limit=5,
            with_payload=default_payload,
            prefetch=None,
        )

        mock_client.query_batch_points.assert_called_once_with(
            collection_name=collection_name, requests=[expected_request]
        )

        assert results == [SearchResponse(id="query_id_1", query="machine learning", matches=[])]

    def test_search_payload_missing_fallbacks(self, mock_client, store, sample_queries):
        mock_api_response = MagicMock()
        mock_api_response.points = [ScoredPoint(id=999, version=1, score=0.71, payload=None)]
        mock_client.query_batch_points.return_value = [mock_api_response]

        config = SearchConfig(mode=SearchMode.DENSE)
        results = list(store.search(sample_queries, filters=None, config=config))

        expected_responses = [
            SearchResponse(
                id="query_id_1",
                query="machine learning",
                matches=[SearchResult(id=999, score=0.71, text="", metadata={})],
            )
        ]
        assert results == expected_responses

    def test_search_engine_length_mismatch_exception(self, mock_client, store, sample_queries):
        mock_client.query_batch_points.return_value = []
        config = SearchConfig(mode=SearchMode.DENSE)

        with pytest.raises(RuntimeError, match="Qdrant returned a different number of responses"):
            list(store.search(sample_queries, filters=None, config=config))
