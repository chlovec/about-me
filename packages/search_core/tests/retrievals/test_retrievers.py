from unittest.mock import MagicMock, call

import numpy as np
import pytest
from search_core.interfaces import DenseEncoder, SparseEncoder, VectorStore
from search_core.models import (
    Document,
    SearchConfig,
    SearchMode,
    SearchQuery,
    SearchResponse,
    SparseVector,
)
from search_core.retrievals import Retriever


@pytest.fixture
def mock_dependencies():
    """Fixture to provide mocked dependencies for the Retriever."""
    dense_mock = MagicMock(spec=DenseEncoder)
    sparse_mock = MagicMock(spec=SparseEncoder)
    store_mock = MagicMock(spec=VectorStore)
    return dense_mock, sparse_mock, store_mock


@pytest.fixture
def retriever(mock_dependencies):
    """Fixture to initialize the Retriever with mocked dependencies."""
    dense_mock, sparse_mock, store_mock = mock_dependencies
    return Retriever(
        dense_model=dense_mock,
        sparse_model=sparse_mock,
        embedding_store=store_mock,
    )


class TestCreateEmbeddingsInternal:
    def test_returns_none_values_if_both_encodings_are_disabled(self, retriever, mock_dependencies):
        """Test that if both dense and sparse are False, both attributes return None."""
        dense_mock, sparse_mock, _ = mock_dependencies

        result = retriever._create_embeddings(texts=["some text"], dense=False, sparse=False)

        assert result.dense is None
        assert result.sparse is None
        dense_mock.encode.assert_not_called()
        sparse_mock.encode.assert_not_called()

    def test_generates_both_dense_and_sparse_embeddings(self, retriever, mock_dependencies):
        """Test the default path where both dense and sparse configurations are enabled."""
        dense_mock, sparse_mock, _ = mock_dependencies
        texts = ["hello world", "pytest testing"]

        # Configure stubs
        mock_dense_output = np.array(
            [[0.1, 0.2], [0.3, 0.4]], dtype="float64"
        )  # Intentional float64 to test casting
        dense_mock.encode.return_value = mock_dense_output

        mock_sparse_vectors = [MagicMock(spec=SparseVector), MagicMock(spec=SparseVector)]
        sparse_mock.encode.return_value = mock_sparse_vectors

        # Act
        result = retriever._create_embeddings(texts, dense=True, sparse=True, show_progress=True)

        # Assert Dense Logic & Parameter Forwarding
        dense_mock.encode.assert_called_once_with(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        assert result.dense.dtype == np.float32  # Verifies the .astype("float32") conversion
        np.testing.assert_array_equal(result.dense, mock_dense_output.astype("float32"))

        # Assert Sparse Logic
        sparse_mock.encode.assert_called_once_with(texts)
        assert result.sparse == mock_sparse_vectors

    def test_dense_only_skips_sparse_encoding(self, retriever, mock_dependencies):
        """Test that configuring sparse=False skips the sparse model altogether."""
        dense_mock, sparse_mock, _ = mock_dependencies
        texts = ["sample text"]

        dense_mock.encode.return_value = np.array([[0.9]], dtype="float32")

        result = retriever._create_embeddings(texts, dense=True, sparse=False)

        assert result.dense is not None
        assert result.sparse is None
        dense_mock.encode.assert_called_once()
        sparse_mock.encode.assert_not_called()

    def test_sparse_only_skips_dense_encoding(self, retriever, mock_dependencies):
        """Test that configuring dense=False skips the dense model altogether."""
        dense_mock, sparse_mock, _ = mock_dependencies
        texts = ["sample text"]

        sparse_mock.encode.return_value = [MagicMock(spec=SparseVector)]

        result = retriever._create_embeddings(texts, dense=False, sparse=True)

        assert result.dense is None
        assert result.sparse is not None
        dense_mock.encode.assert_not_called()
        sparse_mock.encode.assert_called_once_with(texts)


class TestCreateMetadataIndexes:
    def test_success_with_valid_fields(self, retriever, mock_dependencies):
        """Test that unique fields successfully call the embedding store's index method."""
        _, _, store_mock = mock_dependencies
        fields = ["title", "author", "created_at"]

        retriever.create_metadata_indexes(fields)

        assert store_mock.create_metadata_index.call_count == 3
        store_mock.create_metadata_index.assert_has_calls(
            [call("title"), call("author"), call("created_at")], any_order=True
        )

    def test_removes_duplicate_fields(self, retriever, mock_dependencies):
        """Test that duplicate fields are deduplicated and executed only once."""
        _, _, store_mock = mock_dependencies
        fields = ["title", "title", "author", "author", "author"]

        retriever.create_metadata_indexes(fields)

        assert store_mock.create_metadata_index.call_count == 2
        store_mock.create_metadata_index.assert_has_calls(
            [call("title"), call("author")], any_order=True
        )

    def test_raises_value_error_on_empty_list(self, retriever):
        """Test that passing an empty fields list raises a ValueError."""
        with pytest.raises(ValueError, match="Cannot create index on empty fields."):
            retriever.create_metadata_indexes([])

    def test_propagates_store_exceptions(self, retriever, mock_dependencies):
        """Test that exceptions raised by the underlying VectorStore propagate up."""
        _, _, store_mock = mock_dependencies
        fields = ["title"]
        store_mock.create_metadata_index.side_effect = Exception("Database connection failure")

        with pytest.raises(Exception, match="Database connection failure"):
            retriever.create_metadata_indexes(fields)


class TestCreateAndSaveEmbeddings:
    def test_raises_value_error_on_invalid_batch_size(self, retriever):
        """Test that a batch size of 0 or less raises a ValueError."""
        doc = MagicMock(spec=Document)
        generator = retriever.create_and_save_embeddings([doc], batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be > 0"):
            next(generator)

    def test_processes_and_saves_in_batches(self, retriever, mock_dependencies):
        """Test batching pipeline mechanics, mapping behavior, and running totals."""
        dense_mock, sparse_mock, store_mock = mock_dependencies

        # 1. Setup mock documents
        doc1 = MagicMock(spec=Document, text="Text one")
        doc2 = MagicMock(spec=Document, text="Text two")
        doc3 = MagicMock(spec=Document, text="Text three")

        # Stubs for converted values returned by `doc.to_embedded`
        embedded_doc1 = MagicMock()
        embedded_doc2 = MagicMock()
        embedded_doc3 = MagicMock()

        doc1.to_embedded.return_value = embedded_doc1
        doc2.to_embedded.return_value = embedded_doc2
        doc3.to_embedded.return_value = embedded_doc3

        # 2. Setup encoder mock return values
        # Batch 1 (Size 2)
        dense_mock.encode.side_effect = [
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"),  # Batch 1 dense
            np.array([[0.5, 0.6]], dtype="float32"),  # Batch 2 dense
        ]

        # Mock sparse vectors (can be arbitrary types matching your contract)
        sparse_vec1 = MagicMock(spec=SparseVector)
        sparse_vec2 = MagicMock(spec=SparseVector)
        sparse_vec3 = MagicMock(spec=SparseVector)

        sparse_mock.encode.side_effect = [
            [sparse_vec1, sparse_vec2],  # Batch 1 sparse
            [sparse_vec3],  # Batch 2 sparse
        ]

        # 3. Setup store return values (returns the amount of items saved per batch)
        store_mock.save_embeddings.side_effect = [2, 1]

        # 4. Run generator with batch_size=2 on 3 items
        generator = retriever.create_and_save_embeddings(
            documents=[doc1, doc2, doc3],
            create_sparse_embeddings=True,
            wait=True,
            batch_size=2,
            show_progress=False,
        )

        # First Yield: Batch 1 processed (2 docs)
        total_saved_step_1 = next(generator)
        assert total_saved_step_1 == 2
        store_mock.save_embeddings.assert_called_with(
            documents=[embedded_doc1, embedded_doc2], wait=True
        )

        # Second Yield: Batch 2 processed (1 remaining doc)
        total_saved_step_2 = next(generator)
        assert total_saved_step_2 == 3
        store_mock.save_embeddings.assert_called_with(documents=[embedded_doc3], wait=True)

        # Verify execution completed
        with pytest.raises(StopIteration):
            next(generator)

    def test_bypasses_sparse_embeddings(self, retriever, mock_dependencies):
        """Test that if create_sparse_embeddings=False, sparse models aren't called."""
        dense_mock, sparse_mock, store_mock = mock_dependencies

        doc = MagicMock(spec=Document, text="Text")
        dense_mock.encode.return_value = np.array([[0.1]], dtype="float32")
        store_mock.save_embeddings.return_value = 1

        generator = retriever.create_and_save_embeddings(
            documents=[doc], create_sparse_embeddings=False, batch_size=1
        )
        list(generator)  # Exhaust generator to execute logic

        # Verify sparse encoder was never touched
        sparse_mock.encode.assert_not_called()

    def test_raises_value_error_on_empty_documents(self, retriever):
        """Test that passing an empty list of documents raises a ValueError."""
        generator = retriever.create_and_save_embeddings(documents=[], batch_size=32)

        with pytest.raises(ValueError, match="documents cannot be empty"):
            next(generator)

    class TestRetrieveDocuments:
        def test_raises_value_error_on_invalid_batch_size(self, retriever):
            """Test that a batch size of 0 or less raises a ValueError."""
            query = MagicMock(spec=SearchQuery)
            config = MagicMock(spec=SearchConfig)

            generator = retriever.retrieve_documents(
                [query], filters=None, config=config, batch_size=0
            )

            with pytest.raises(ValueError, match="batch_size must be > 0"):
                next(generator)

        def test_hybrid_mode_generates_both_embeddings(self, retriever, mock_dependencies):
            """Test that SearchMode.HYBRID triggers both dense and sparse encoders."""
            dense_mock, sparse_mock, store_mock = mock_dependencies

            # 1. Setup mock query and config
            query = MagicMock(spec=SearchQuery, text="hybrid search query")
            config = MagicMock(spec=SearchConfig, mode=SearchMode.HYBRID)
            filters = {"category": "test"}

            # Stubs for conversions and final response
            embedded_query = MagicMock()
            mock_response = MagicMock(spec=SearchResponse)

            query.to_embedded.return_value = embedded_query
            store_mock.search.return_value = mock_response

            # Stub encoder returns
            dense_mock.encode.return_value = np.array([[0.1, 0.2]], dtype="float32")
            sparse_vec = MagicMock(spec=SparseVector)
            sparse_mock.encode.return_value = [sparse_vec]

            # 2. Act
            generator = retriever.retrieve_documents(
                [query], filters=filters, config=config, batch_size=1
            )
            results = list(generator)

            # 3. Assert
            assert results == [mock_response]

            # Verify internal encoders were both called
            dense_mock.encode.assert_called_once_with(
                ["hybrid search query"],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            sparse_mock.encode.assert_called_once_with(["hybrid search query"])

            # Verify conversion mapped both mock outputs
            query.to_embedded.assert_called_once()

            # Verify store payload
            store_mock.search.assert_called_once_with([embedded_query], filters, config)

        def test_non_hybrid_mode_skips_sparse_embeddings(self, retriever, mock_dependencies):
            """Test that non-hybrid modes (e.g., DENSE) skip sparse encoding completely."""
            dense_mock, sparse_mock, store_mock = mock_dependencies

            query = MagicMock(spec=SearchQuery, text="dense search query")
            config = MagicMock(spec=SearchConfig, mode=SearchMode.DENSE)  # Not HYBRID

            dense_mock.encode.return_value = np.array([[0.1, 0.2]], dtype="float32")

            generator = retriever.retrieve_documents(
                [query], filters=None, config=config, batch_size=1
            )
            list(generator)  # Exhaust iterator

            # Verify dense was called but sparse was cleanly bypassed
            dense_mock.encode.assert_called_once()
            sparse_mock.encode.assert_not_called()

        def test_processes_queries_in_batches(self, retriever, mock_dependencies):
            """Test that queries are batched correctly and yielded incrementally."""
            _, _, store_mock = mock_dependencies

            q1 = MagicMock(spec=SearchQuery, text="one")
            q2 = MagicMock(spec=SearchQuery, text="two")
            q3 = MagicMock(spec=SearchQuery, text="three")
            config = MagicMock(spec=SearchConfig, mode=SearchMode.DENSE)

            response_batch_1 = MagicMock(spec=SearchResponse)
            response_batch_2 = MagicMock(spec=SearchResponse)
            store_mock.search.side_effect = [response_batch_1, response_batch_2]

            generator = retriever.retrieve_documents(
                [q1, q2, q3], filters=None, config=config, batch_size=2
            )

            # First yield should process batch 1 (size 2)
            assert next(generator) == response_batch_1
            assert store_mock.search.call_count == 1

            # Second yield should process batch 2 (size 1 remaining)
            assert next(generator) == response_batch_2
            assert store_mock.search.call_count == 2

            with pytest.raises(StopIteration):
                next(generator)
