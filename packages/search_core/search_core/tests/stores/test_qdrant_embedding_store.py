import pytest

from unittest.mock import MagicMock

# from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector as QdrantSparseVector,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from search_core import QdrantEmbeddingStore


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.collection_exists.return_value = True
    return client

@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    # Default to collection not existing to hit creation logic
    client.collection_exists.return_value = False
    return client

class TestQdrantEmbeddingStoreInit:
    def test_init_creates_collection_if_not_exists(self, mock_qdrant_client):
        mock_qdrant_client.collection_exists.return_value = False
        
        QdrantEmbeddingStore(
            client=mock_qdrant_client,
            collection_name="test_collection",
            vector_size=128,
            distance=Distance.COSINE
        )
        
        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
        mock_qdrant_client.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vectors_config={"dense": VectorParams(size=128, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(modifier=None)}
        )
        
    def test_init_skips_creation_if_collection_exists(self, mock_qdrant_client):
            mock_qdrant_client.collection_exists.return_value = True
            
            QdrantEmbeddingStore(
                client=mock_qdrant_client,
                collection_name="test_collection",
                vector_size=128
            )
            
            mock_qdrant_client.create_collection.assert_not_called()
            
    def test_init_raises_unexpected_response_if_generic_error(self, mock_qdrant_client):
        mock_qdrant_client.collection_exists.return_value = False
        
        expected_message = "Something went entirely wrong"
        mock_qdrant_client.create_collection.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Error",
            content=b"Internal Server Error Context",
            headers={"X-Test-Header": "error"}
        )
        
        # Capture the exception context using 'as exc_info'
        with pytest.raises(UnexpectedResponse) as exc_info:
            QdrantEmbeddingStore(
                client=mock_qdrant_client,
                collection_name="test_collection",
                vector_size=128
            )
        
        # Verify the exception contents explicitly
        assert exc_info.value.status_code == 500
        assert exc_info.value.reason_phrase == "Internal Error"
        assert exc_info.value.headers == {"X-Test-Header": "error"}
        assert exc_info.value.content == b"Internal Server Error Context"
        