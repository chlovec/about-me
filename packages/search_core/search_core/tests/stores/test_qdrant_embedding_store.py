import pytest

from unittest.mock import MagicMock, call, patch

from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector as QdrantSparseVector,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from search_core import (
    QdrantEmbeddingStore,
    QdrantStoreConfig,
)


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.collection_exists.return_value = False
    return client


@pytest.fixture
def store_instance():
    """Provides a QdrantEmbeddingStore instance with a mocked client."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    return QdrantEmbeddingStore(
        client=mock_client,
        collection_name="test_collection",
        vector_size=128,
        max_workers=4,
    )

class TestQdrantEmbeddingStoreConnect:

    @patch("search_core.stores.qdrant_embedding_store.QdrantClient")
    @patch("search_core.stores.qdrant_embedding_store.QdrantEmbeddingStore.__init__", return_value=None)
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
            host=None,      # Should be filtered out
            port=None,      # Should be filtered out
            max_workers=32,
            client_kwargs={"timeout": 60, "grpc_port": 6334}
        )

        # Act
        store = QdrantEmbeddingStore.connect(config)

        # Assert: QdrantClient was initialized only with non-None variables merged with client_kwargs
        mock_qdrant_client_cls.assert_called_once_with(
            url="https://localhost:6333",
            api_key="super-secret-key",
            timeout=60,
            grpc_port=6334
        )

        # Assert: The class constructor (__init__) received the generated client and config properties
        mock_init.assert_called_once_with(
            client=mock_qdrant_client_cls.return_value,
            collection_name="production_collection",
            vector_size=512,
            distance=Distance.DOT,
            max_workers=32
        )
        
        assert isinstance(store, QdrantEmbeddingStore)

    @patch("search_core.stores.qdrant_embedding_store.QdrantClient")
    @patch("search_core.stores.qdrant_embedding_store.QdrantEmbeddingStore.__init__", return_value=None)
    def test_connect_handles_minimal_config(self, mock_init, mock_qdrant_client_cls):
        """Verifies connect behavior when only mandatory config properties are provided."""
        config = QdrantStoreConfig(
            collection_name="minimal_collection",
            vector_size=128
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
            max_workers=16            # Default from QdrantStoreConfig
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
        
        
    def test_init_handles_race_condition_cleanly_when_already_exists(self, mock_qdrant_client):
        # Simulate the specific cluster race condition error string
        mock_exception = UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=b"Collection test_collection already exists!",
            headers={}
        )
        mock_qdrant_client.create_collection.side_effect = mock_exception

        # Act & Assert: This should NOT raise an exception and should complete initialization
        store = QdrantEmbeddingStore(
            client=mock_qdrant_client,
            collection_name="test_collection",
            vector_size=128
        )
        
        # Verify it passed line 98 and successfully set up the store instance
        assert store.collection_name == "test_collection"


class TestCreateMetadataIndexes:

    def test_create_metadata_indexes_success(self, store_instance):
        """Verifies that payload indexes are created for every provided field."""
        fields_to_index = ["user_id", "status", "tenant_id"]

        # Act
        store_instance.create_metadata_indexes(fields_to_index)

        # Assert each expected payload index call occurred
        expected_calls = [
            call(
                collection_name="test_collection",
                field_name="metadata.user_id",
                field_schema="keyword",
            ),
            call(
                collection_name="test_collection",
                field_name="metadata.status",
                field_schema="keyword",
            ),
            call(
                collection_name="test_collection",
                field_name="metadata.tenant_id",
                field_schema="keyword",
            ),
        ]
        store_instance.client.create_payload_index.assert_has_calls(
            expected_calls, any_order=True
        )
        assert store_instance.client.create_payload_index.call_count == 3

    @pytest.mark.parametrize("empty_input", [[]])
    def test_create_metadata_indexes_raises_value_error_if_empty(
        self, store_instance, empty_input
    ):
        """Verifies that a ValueError is raised when an empty fields list is provided."""

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            store_instance.create_metadata_indexes(empty_input)

        # Confirm exception context details
        assert "fields' list cannot be empty" in str(exc_info.value)
        # Verify the underlying execution was halted immediately and no indexing client was called
        store_instance.client.create_payload_index.assert_not_called()
