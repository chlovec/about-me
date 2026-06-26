import pytest

from unittest.mock import MagicMock, call, patch

import numpy as np

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchExcept,
    MatchValue,
    PointStruct,
    QueryRequest,
    Range,
    Rrf,
    RrfQuery,
    ScoredPoint,
    UpdateResult,
    UpdateStatus,
    VectorParams,
    SparseVectorParams,
    SparseVector as QdrantSparseVector,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from search_core import (
    EmbeddedDocument,
    EmbeddedQuery,
    QdrantEmbeddingStore,
    QdrantStoreConfig,
    SearchConfig,
    SearchMode,
    SearchResponse,
    SearchResult,
)


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.collection_exists.return_value = False
    return client


@pytest.fixture
def store_instance(mock_qdrant_client):
    """Provides a QdrantEmbeddingStore instance with a fixed vector size configuration."""
    return QdrantEmbeddingStore(
        client=mock_qdrant_client,
        collection_name="test_collection",
        vector_size=3,
    )


class TestQdrantEmbeddingStoreConnect:

    @patch("search_core.stores.qdrant_embedding_store.QdrantClient")
    @patch(
        "search_core.stores.qdrant_embedding_store.QdrantEmbeddingStore.__init__",
        return_value=None,
    )
    def test_connect_filters_none_and_unpacks_kwargs(
        self, mock_init, mock_qdrant_client_cls
    ):
        """Verifies that connect filters out None parameters, merges client_kwargs,

        and properly invokes the QdrantClient constructor and internal __init__.
        """
        # Create a config with mixed configurations, some Nones, and custom escape-hatch kwargs
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

        # Act
        store = QdrantEmbeddingStore.connect(config)

        # Assert: QdrantClient was initialized only with non-None variables merged with client_kwargs
        mock_qdrant_client_cls.assert_called_once_with(
            url="https://localhost:6333",
            api_key="super-secret-key",
            timeout=60,
            grpc_port=6334,
        )

        # Assert: The class constructor (__init__) received the generated client and config properties
        mock_init.assert_called_once_with(
            client=mock_qdrant_client_cls.return_value,
            collection_name="production_collection",
            vector_size=512,
            distance=Distance.DOT,
        )

        assert isinstance(store, QdrantEmbeddingStore)

    @patch("search_core.stores.qdrant_embedding_store.QdrantClient")
    @patch(
        "search_core.stores.qdrant_embedding_store.QdrantEmbeddingStore.__init__",
        return_value=None,
    )
    def test_connect_handles_minimal_config(self, mock_init, mock_qdrant_client_cls):
        """Verifies connect behavior when only mandatory config properties are provided."""
        config = QdrantStoreConfig(
            collection_name="minimal_collection", vector_size=128
        )

        # Act
        QdrantEmbeddingStore.connect(config)

        # Assert: Client constructor should have been called completely empty (defaults)
        mock_qdrant_client_cls.assert_called_once_with()

        # Assert: Class instantiation forwarded the system defaults perfectly
        mock_init.assert_called_once_with(
            client=mock_qdrant_client_cls.return_value,
            collection_name="minimal_collection",
            vector_size=128,
            distance=Distance.COSINE,  # Default from QdrantStoreConfig
        )


class TestQdrantEmbeddingStoreInit:
    def test_init_creates_collection_if_not_exists(self, mock_qdrant_client):
        QdrantEmbeddingStore(
            client=mock_qdrant_client,
            collection_name="test_collection",
            vector_size=128,
            distance=Distance.COSINE,
        )

        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
        mock_qdrant_client.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vectors_config={"dense": VectorParams(size=128, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(modifier=None)},
        )

    def test_init_skips_creation_if_collection_exists(self, mock_qdrant_client):
        mock_qdrant_client.collection_exists.return_value = True
        QdrantEmbeddingStore(
            client=mock_qdrant_client,
            collection_name="test_collection",
            vector_size=128,
        )

        mock_qdrant_client.create_collection.assert_not_called()

    def test_init_raises_unexpected_response_if_generic_error(self, mock_qdrant_client):
        mock_qdrant_client.create_collection.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Error",
            content=b"Internal Server Error Context",
            headers={"X-Test-Header": "error"},
        )

        # Capture the exception context using 'as exc_info'
        with pytest.raises(UnexpectedResponse) as exc_info:
            QdrantEmbeddingStore(
                client=mock_qdrant_client,
                collection_name="test_collection",
                vector_size=128,
            )

        # Verify the exception contents explicitly
        assert exc_info.value.status_code == 500
        assert exc_info.value.reason_phrase == "Internal Error"
        assert exc_info.value.headers == {"X-Test-Header": "error"}
        assert exc_info.value.content == b"Internal Server Error Context"

    def test_init_handles_race_condition_cleanly_when_already_exists(
        self, mock_qdrant_client
    ):
        # Simulate the specific cluster race condition error string
        mock_exception = UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=b"Collection test_collection already exists!",
            headers={},
        )
        mock_qdrant_client.create_collection.side_effect = mock_exception

        # Act & Assert: This should NOT raise an exception and should complete initialization
        store = QdrantEmbeddingStore(
            client=mock_qdrant_client,
            collection_name="test_collection",
            vector_size=128,
        )

        # Verify it passed line 98 and successfully set up the store instance
        assert store.collection_name == "test_collection"


class TestCreateMetadataIndexes:

    def test_create_metadata_indexes_success(self, store_instance):
        """Verifies that payload index is created for the provided field."""

        # Act
        store_instance.create_metadata_indexes("user_name")

        # Assert the expected payload index call occurred
        store_instance.client.create_payload_index.assert_called_once_with(
            collection_name="test_collection",
            field_name="metadata.user_name",
            field_schema="keyword",
        )


class TestSaveEmbeddings:

    def test_save_embeddings_dense_only_wait_true(self, store_instance):
        """Verifies successful upsert of dense-only documents with wait=True."""
        # Arrange
        mock_doc = MagicMock(spec=EmbeddedDocument)
        mock_doc.id = 101
        mock_doc.text = "Sample text payload"
        mock_doc.embedding = np.array([0.1, 0.2, 0.3])
        mock_doc.sparse_vector = None
        mock_doc.metadata = {"category": "test"}

        documents = [mock_doc]

        # Mock Qdrant upsert response matching wait=True expectation
        store_instance.client.upsert.return_value = UpdateResult(
            operation_id=1, status=UpdateStatus.COMPLETED
        )

        # Act
        result_count = store_instance.save_embeddings(documents, wait=True)

        # Assert
        assert result_count == 1
        mock_doc.validate.assert_called_once_with(store_instance.vector_size)

        # Verify the structure passed to QdrantClient
        store_instance.client.upsert.assert_called_once_with(
            collection_name="test_collection",
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

    def test_save_embeddings_hybrid_wait_false(self, store_instance):
        """Verifies successful upsert of hybrid (dense + sparse) documents with wait=False."""
        # Arrange
        mock_doc = MagicMock(spec=EmbeddedDocument)
        mock_doc.id = 202
        mock_doc.text = "Hybrid vector text"
        mock_doc.embedding = np.array([0.9, 0.8, 0.7])

        # Construct sparse vector properties
        mock_sparse = MagicMock()
        mock_sparse.indices = [12, 45]
        mock_sparse.values = [0.35, 0.88]
        mock_doc.sparse_vector = mock_sparse
        mock_doc.metadata = {}

        documents = [mock_doc]

        # Mock Qdrant upsert response matching wait=False expectation
        store_instance.client.upsert.return_value = UpdateResult(
            operation_id=2, status=UpdateStatus.ACKNOWLEDGED
        )

        # Act
        result_count = store_instance.save_embeddings(documents, wait=False)

        # Assert
        assert result_count == 1
        mock_doc.validate.assert_called_once_with(store_instance.vector_size)

        # Verify sparse mapping logic transformation
        store_instance.client.upsert.assert_called_once_with(
            collection_name="test_collection",
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
        self, store_instance, wait_param, returned_status
    ):
        """Ensures a RuntimeError is thrown if Qdrant reports a status mismatch."""
        # Arrange
        mock_doc = MagicMock(spec=EmbeddedDocument)
        mock_doc.id = 500
        mock_doc.text = "Error vector text"
        mock_doc.embedding = np.array([0.0, 0.0, 0.0])
        mock_doc.sparse_vector = None
        mock_doc.metadata = {}

        store_instance.client.upsert.return_value = UpdateResult(
            operation_id=99, status=returned_status
        )

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            store_instance.save_embeddings([mock_doc], wait=wait_param)

        # Explicitly verify the error format matches the implementation message
        assert f"Upsert failed with status={returned_status}" in str(exc_info.value)


class TestQdrantStoreSearch:

    def test_search_dense_mode_success(self, store_instance):
        """Verifies correct structural assembly and execution of a dense-only batch query."""
        # Arrange
        queries = [
            EmbeddedQuery(
                id="q-1", embedding=np.array([0.1, 0.2, 0.3]), sparse_vector=None
            )
        ]
        filters = {"category": "finance"}
        config = SearchConfig(mode=SearchMode.DENSE, k=5, return_metadata=["tenant_id"])

        # Mock individual point score results returned by Qdrant
        mock_point = ScoredPoint(
            id=1001,
            version=1,
            score=0.89,
            payload={
                "text": "Dense response text matching query",
                "doc_id": "doc-1001",
                "metadata": {"tenant_id": "abc-123"},
            },
        )
        mock_batch_response = MagicMock()
        mock_batch_response.points = [mock_point]

        # Ensure batch returns matching request structures length
        store_instance.client.query_batch_points.return_value = [mock_batch_response]

        # Act
        results = list(
            store_instance.search(queries=queries, filters=filters, config=config)
        )

        # Assert
        assert len(results) == 1
        response = results[0]
        assert response.id == "q-1"
        assert len(response.matches) == 1

        match = response.matches[0]
        assert match.id == "doc-1001"
        assert match.text == "Dense response text matching query"
        assert match.score == 0.89
        assert match.metadata == {"tenant_id": "abc-123"}

        # Verify underlying query request validation parameters
        store_instance.client.query_batch_points.assert_called_once()
        _, kwargs = store_instance.client.query_batch_points.call_args
        assert kwargs["collection_name"] == "test_collection"

        requests = kwargs["requests"]
        assert len(requests) == 1
        assert isinstance(requests[0], QueryRequest)
        assert requests[0].using == "dense"
        assert requests[0].limit == 5
        assert requests[0].with_payload == ["text", "doc_id", "metadata.tenant_id"]

    def test_search_hybrid_rrf_mode_success(self, store_instance):
        """Verifies that combined dense and sparse configurations trigger multi-prefetch RRF structures."""
        # Arrange
        mock_sparse = MagicMock()
        mock_sparse.indices = [2, 8]
        mock_sparse.values = [0.4, 0.7]

        queries = [
            EmbeddedQuery(
                id="q-2", embedding=np.array([0.5, 0.5, 0.5]), sparse_vector=mock_sparse
            )
        ]

        config = SearchConfig(
            mode=SearchMode.HYBRID,  # Assuming SearchMode supports HYBRID or equivalent non-DENSE state
            k=3,
            prefetch_k=20,
            rrf_k=50,
            return_metadata=[],
        )

        mock_batch_response = MagicMock(points=[])
        store_instance.client.query_batch_points.return_value = [mock_batch_response]

        # Act
        list(store_instance.search(queries=queries, filters=None, config=config))

        # Assert
        store_instance.client.query_batch_points.assert_called_once()
        _, kwargs = store_instance.client.query_batch_points.call_args
        requests = kwargs["requests"]

        req = requests[0]
        assert req.prefetch is not None
        assert len(req.prefetch) == 2

        # Validate Dense Prefetch
        assert req.prefetch[0].using == "dense"
        assert req.prefetch[0].limit == 20

        # Validate Sparse Prefetch
        assert req.prefetch[1].using == "sparse"
        assert isinstance(req.prefetch[1].query, QdrantSparseVector)
        assert req.prefetch[1].query.indices == [2, 8]
        assert req.prefetch[1].query.values == [0.4, 0.7]

        # Validate RRF parameters
        assert isinstance(req.query, RrfQuery)
        assert req.query.rrf == Rrf(k=50)
        assert req.limit == 3

    def test_search_mismatched_response_count_raises_runtime_error(
        self, store_instance
    ):
        """Ensures that if the internal response array size deviates from the submitted batch size,

        a RuntimeError terminates processing instantly.
        """
        # Arrange
        queries = [
            EmbeddedQuery(id="q-1", embedding=np.array([0.1, 0.2, 0.3])),
            EmbeddedQuery(id="q-2", embedding=np.array([0.4, 0.5, 0.6])),
        ]
        config = SearchConfig(mode=SearchMode.DENSE, k=5)

        # Force a length anomaly (2 queries vs 1 response block returned)
        store_instance.client.query_batch_points.return_value = [MagicMock()]

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            list(store_instance.search(queries=queries, filters=None, config=config))

        assert "Qdrant returned a different number of responses than requests" in str(
            exc_info.value
        )

    def test_parse_filters_implicit_in_list_tuple(self, store_instance):
        filters = {"categories": ["finance", "tech"], "tags": ("internal", "verified")}

        parsed = store_instance._parse_filters(filters)

        assert isinstance(parsed, Filter)
        assert len(parsed.must) == 2

        # Verify both conditions transformed into MatchAny entries
        for condition in parsed.must:
            assert isinstance(condition, FieldCondition)
            assert isinstance(condition.match, MatchAny)
            if condition.key == "metadata.categories":
                assert condition.match.any == ["finance", "tech"]
            else:
                assert condition.key == "metadata.tags"
                assert condition.match.any == ["internal", "verified"]

    def test_parse_filters_all_range_operators(self, store_instance):
        filters = {"created_at": {"$gt": 100, "$gte": 101, "$lt": 200, "$lte": 201}}

        parsed = store_instance._parse_filters(filters)

        assert isinstance(parsed, Filter)
        assert len(parsed.must) == 1

        condition = parsed.must[0]
        assert isinstance(condition, FieldCondition)
        assert condition.key == "metadata.created_at"
        assert isinstance(condition.range, Range)
        assert condition.range.gt == 100
        assert condition.range.gte == 101
        assert condition.range.lt == 200
        assert condition.range.lte == 201

    # def test_parse_filters_exclusion_operators(self, store_instance):
    #     filters = {
    #         "status": {"$ne": "archived"},
    #         "team_id": {"$nin": ["team_a", "team_b"]}
    #     }

    #     parsed = store_instance._parse_filters(filters)

    #     assert isinstance(parsed, Filter)
    #     assert len(parsed.must) == 2

    #     for condition in parsed.must:
    #         assert isinstance(condition, FieldCondition)
    #         assert isinstance(condition.match, MatchExcept)
    #         if condition.key == "metadata.status":
    #             assert getattr(condition.match, "except") == ["archived"]
    #         else:
    #             assert condition.key == "metadata.team_id"
    #             assert condition.match.any == ["team_a", "team_b"]

    def test_parse_filters_explicit_equality_operators(self, store_instance):
        filters = {
            "visibility": {"$eq": "public"},
            "region": {"$in": ["us-east", "us-west"]},
        }

        parsed = store_instance._parse_filters(filters)

        assert isinstance(parsed, Filter)
        assert len(parsed.must) == 2

        for condition in parsed.must:
            assert isinstance(condition, FieldCondition)
            if condition.key == "metadata.visibility":
                assert isinstance(condition.match, MatchValue)
                assert condition.match.value == "public"
            else:
                assert condition.key == "metadata.region"
                assert isinstance(condition.match, MatchAny)
                assert condition.match.any == ["us-east", "us-west"]

    def test_parse_filters_exclusion_operators(self, store_instance):
        """Covers lines 157-164: Exclusion operators ($ne, $nin)."""
        from qdrant_client.models import MatchExcept

        filters = {
            "status": {"$ne": "archived"},
            "team_id": {"$nin": ["team_a", "team_b"]},
        }

        parsed = store_instance._parse_filters(filters)

        assert isinstance(parsed, Filter)
        assert len(parsed.must) == 2

        for condition in parsed.must:
            assert isinstance(condition, FieldCondition)
            assert isinstance(condition.match, MatchExcept)

            if condition.key == "metadata.status":
                assert condition.match.except_ == [
                    "archived"
                ]  # Qdrant Client uses trailing underscore for fields mapping to keywords
            else:
                assert condition.key == "metadata.team_id"
                assert condition.match.except_ == ["team_a", "team_b"]
