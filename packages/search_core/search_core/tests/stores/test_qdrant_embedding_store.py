from unittest import TestCase
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
    QueryRequest,
    Range,
    Rrf,
    RrfQuery,
    ScoredPoint,
    SparseVectorParams,
    UpdateResult,
    UpdateStatus,
    VectorParams,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)

from search_core import (
    EmbeddedDocument,
    EmbeddedQuery,
    QdrantEmbeddingStore,
    QdrantStoreConfig,
    SearchConfig,
    SearchMode,
)

# Shared assertion runner without modifying class inheritance
tc = TestCase()


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
    def test_connect_filters_none_and_unpacks_kwargs(self, mock_init, mock_qdrant_client_cls):
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

        tc.assertIsInstance(store, QdrantEmbeddingStore)

    @patch("search_core.stores.qdrant_embedding_store.QdrantClient")
    @patch(
        "search_core.stores.qdrant_embedding_store.QdrantEmbeddingStore.__init__",
        return_value=None,
    )
    def test_connect_handles_minimal_config(self, mock_init, mock_qdrant_client_cls):
        """Verifies connect behavior when only mandatory config properties are provided."""
        config = QdrantStoreConfig(collection_name="minimal_collection", vector_size=128)

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
        tc.assertEqual(exc_info.value.status_code, 500)
        tc.assertEqual(exc_info.value.reason_phrase, "Internal Error")
        tc.assertEqual(exc_info.value.headers, {"X-Test-Header": "error"})
        tc.assertEqual(exc_info.value.content, b"Internal Server Error Context")

    def test_init_handles_race_condition_cleanly_when_already_exists(self, mock_qdrant_client):
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
        tc.assertEqual(store.collection_name, "test_collection")


class TestCreateMetadataIndexes:
    def test_create_metadata_indexes_success(self, store_instance):
        """Verifies that payload index is created for the provided field."""

        # Act
        store_instance.create_metadata_index("user_name")

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
        tc.assertEqual(result_count, 1)
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
        tc.assertEqual(result_count, 1)
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
        tc.assertIn(f"Upsert failed with status={returned_status}", str(exc_info.value))


class TestQdrantStoreSearch:
    @pytest.fixture(autouse=True)
    def setup_default_mock_response(self, store_instance):
        """Helper fixture to prevent boilerplate code.

        Ensures search() always receives a valid, empty mock response list
        unless overridden explicitly by a specific test layout.
        """
        mock_batch_response = MagicMock(points=[])
        store_instance.client.query_batch_points.return_value = [mock_batch_response]

    def test_search_dense_mode_success(self, store_instance):
        """Verifies correct structural assembly and execution of a dense-only batch query."""
        # Arrange
        queries = [
            EmbeddedQuery(
                id="q-1", text="text-query", embedding=np.array([0.1, 0.2, 0.3]), sparse_vector=None
            )
        ]
        filters = {"category": "finance"}
        config = SearchConfig(mode=SearchMode.DENSE, k=5, return_metadata=["tenant_id"])

        # Override mock default to return a rich populated point
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
        store_instance.client.query_batch_points.return_value = [mock_batch_response]

        # Act
        results = list(store_instance.search(queries=queries, filters=filters, config=config))

        # Assert
        tc.assertEqual(len(results), 1)
        response = results[0]
        tc.assertEqual(response.id, "q-1")
        tc.assertEqual(len(response.matches), 1)

        match = response.matches[0]
        tc.assertEqual(match.id, "doc-1001")
        tc.assertEqual(match.text, "Dense response text matching query")
        tc.assertEqual(match.score, 0.89)
        tc.assertEqual(match.metadata, {"tenant_id": "abc-123"})

        store_instance.client.query_batch_points.assert_called_once()
        _, kwargs = store_instance.client.query_batch_points.call_args
        tc.assertEqual(kwargs["collection_name"], "test_collection")

        requests = kwargs["requests"]
        tc.assertEqual(len(requests), 1)
        tc.assertIsInstance(requests[0], QueryRequest)
        tc.assertEqual(requests[0].using, "dense")
        tc.assertEqual(requests[0].limit, 5)
        tc.assertEqual(requests[0].with_payload, ["text", "doc_id", "metadata.tenant_id"])

        # Structural Verification of Implicit Match Filter derived through integration
        filter_obj = requests[0].filter
        tc.assertIsInstance(filter_obj, Filter)
        tc.assertEqual(len(filter_obj.must), 1)
        tc.assertEqual(filter_obj.must[0].key, "metadata.category")
        tc.assertIsInstance(filter_obj.must[0].match, MatchValue)
        tc.assertEqual(filter_obj.must[0].match.value, "finance")

    def test_search_hybrid_rrf_mode_success(self, store_instance):
        """Verifies that combined dense and sparse configurations trigger multi-prefetch RRF structures."""
        # Arrange
        mock_sparse = MagicMock()
        mock_sparse.indices = [2, 8]
        mock_sparse.values = [0.4, 0.7]

        queries = [
            EmbeddedQuery(
                id="q-2",
                text="text-query",
                embedding=np.array([0.5, 0.5, 0.5]),
                sparse_vector=mock_sparse,
            )
        ]
        config = SearchConfig(
            mode=SearchMode.HYBRID, k=3, prefetch_k=20, rrf_k=50, return_metadata=[]
        )

        # Act
        list(store_instance.search(queries=queries, filters=None, config=config))

        # Assert
        store_instance.client.query_batch_points.assert_called_once()
        _, kwargs = store_instance.client.query_batch_points.call_args
        requests = kwargs["requests"]

        req = requests[0]
        tc.assertIsNotNone(req.prefetch)
        tc.assertEqual(len(req.prefetch), 2)

        tc.assertEqual(req.prefetch[0].using, "dense")
        tc.assertEqual(req.prefetch[0].limit, 20)

        tc.assertEqual(req.prefetch[1].using, "sparse")
        tc.assertIsInstance(req.prefetch[1].query, QdrantSparseVector)
        tc.assertEqual(req.prefetch[1].query.indices, [2, 8])
        tc.assertEqual(req.prefetch[1].query.values, [0.4, 0.7])

        tc.assertIsInstance(req.query, RrfQuery)
        tc.assertEqual(req.query.rrf, Rrf(k=50))
        tc.assertEqual(req.limit, 3)

    def test_search_mismatched_response_count_raises_runtime_error(self, store_instance):
        """Ensures that if the internal response array size deviates from the submitted batch size,

        a RuntimeError terminates processing instantly.
        """
        # Arrange
        queries = [
            EmbeddedQuery(id="q-1", text="text-query-1", embedding=np.array([0.1, 0.2, 0.3])),
            EmbeddedQuery(id="q-2", text="text-query-2", embedding=np.array([0.4, 0.5, 0.6])),
        ]
        config = SearchConfig(mode=SearchMode.DENSE, k=5)

        # Break expectation sequence manually
        store_instance.client.query_batch_points.return_value = [MagicMock()]

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            list(store_instance.search(queries=queries, filters=None, config=config))

        tc.assertIn(
            "Qdrant returned a different number of responses than requests",
            str(exc_info.value),
        )

    def test_search_with_implicit_in_list_tuple_filters(self, store_instance):
        """Validates evaluation pipelines transforming simple list structures into MatchAny conditions."""
        queries = [EmbeddedQuery(id="q-1", text="text-query", embedding=np.array([0.1, 0.2, 0.3]))]
        filters = {"categories": ["finance", "tech"], "tags": ("internal", "verified")}
        config = SearchConfig(mode=SearchMode.DENSE, k=5)

        list(store_instance.search(queries=queries, filters=filters, config=config))

        _, kwargs = store_instance.client.query_batch_points.call_args
        parsed_filter = kwargs["requests"][0].filter

        tc.assertEqual(len(parsed_filter.must), 2)
        for condition in parsed_filter.must:
            tc.assertIsInstance(condition, FieldCondition)
            tc.assertIsInstance(condition.match, MatchAny)
            if condition.key == "metadata.categories":
                tc.assertEqual(condition.match.any, ["finance", "tech"])
            else:
                tc.assertEqual(condition.key, "metadata.tags")
                tc.assertEqual(condition.match.any, ["internal", "verified"])

    def test_search_with_all_range_operator_filters(self, store_instance):
        """Validates evaluation blocks mapping numeric sub-bounds down into isolated Qdrant Range records."""
        queries = [EmbeddedQuery(id="q-1", text="text-query", embedding=np.array([0.1, 0.2, 0.3]))]
        filters = {"created_at": {"$gt": 100, "$gte": 101, "$lt": 200, "$lte": 201}}
        config = SearchConfig(mode=SearchMode.DENSE, k=5)

        list(store_instance.search(queries=queries, filters=filters, config=config))

        _, kwargs = store_instance.client.query_batch_points.call_args
        parsed_filter = kwargs["requests"][0].filter

        tc.assertEqual(len(parsed_filter.must), 1)
        condition = parsed_filter.must[0]
        tc.assertEqual(condition.key, "metadata.created_at")
        tc.assertIsInstance(condition.range, Range)
        tc.assertEqual(condition.range.gt, 100)
        tc.assertEqual(condition.range.gte, 101)
        tc.assertEqual(condition.range.lt, 200)
        tc.assertEqual(condition.range.lte, 201)

    def test_search_with_explicit_equality_operators(self, store_instance):
        """Validates translation logic tracking exact match definitions ($eq, $in)."""
        queries = [EmbeddedQuery(id="q-1", text="text-query", embedding=np.array([0.1, 0.2, 0.3]))]
        filters = {
            "visibility": {"$eq": "public"},
            "region": {"$in": ["us-east", "us-west"]},
        }
        config = SearchConfig(mode=SearchMode.DENSE, k=5)

        list(store_instance.search(queries=queries, filters=filters, config=config))

        _, kwargs = store_instance.client.query_batch_points.call_args
        parsed_filter = kwargs["requests"][0].filter

        tc.assertEqual(len(parsed_filter.must), 2)
        for condition in parsed_filter.must:
            if condition.key == "metadata.visibility":
                tc.assertIsInstance(condition.match, MatchValue)
                tc.assertEqual(condition.match.value, "public")
            else:
                tc.assertEqual(condition.key, "metadata.region")
                tc.assertIsInstance(condition.match, MatchAny)
                tc.assertEqual(condition.match.any, ["us-east", "us-west"])

    def test_search_with_exclusion_operators(self, store_instance):
        """Validates translation blocks handling inversions ($ne, $nin)."""
        queries = [EmbeddedQuery(id="q-1", text="text-query", embedding=np.array([0.1, 0.2, 0.3]))]
        filters = {
            "status": {"$ne": "archived"},
            "team_id": {"$nin": ["team_a", "team_b"]},
        }
        config = SearchConfig(mode=SearchMode.DENSE, k=5)

        list(store_instance.search(queries=queries, filters=filters, config=config))

        _, kwargs = store_instance.client.query_batch_points.call_args
        parsed_filter = kwargs["requests"][0].filter

        tc.assertEqual(len(parsed_filter.must), 2)
        for condition in parsed_filter.must:
            tc.assertIsInstance(condition.match, MatchExcept)
            if condition.key == "metadata.status":
                tc.assertEqual(condition.match.except_, ["archived"])
            else:
                tc.assertEqual(condition.key, "metadata.team_id")
                tc.assertEqual(condition.match.except_, ["team_a", "team_b"])

    def test_search_with_unknown_operator_clears_branch_fall_through(self, store_instance):
        """Covers branch gap 166->157 via the public interface.

        Injecting an unrecognized nested operator payload ensures that execution paths
        pass through the final logical check and return safely to the loop initialization structure.
        """
        queries = [EmbeddedQuery(id="q-1", text="text-query", embedding=np.array([0.1, 0.2, 0.3]))]
        filters = {"price": {"$unknown": "value"}}
        config = SearchConfig(mode=SearchMode.DENSE, k=5)

        list(store_instance.search(queries=queries, filters=filters, config=config))

        _, kwargs = store_instance.client.query_batch_points.call_args
        parsed_filter = kwargs["requests"][0].filter

        # The unknown key should be skipped entirely, producing an empty list of must requirements
        tc.assertEqual(len(parsed_filter.must), 0)
