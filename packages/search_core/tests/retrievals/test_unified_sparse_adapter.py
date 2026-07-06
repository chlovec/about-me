from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest
import search_core.retrievals.retrievers as target_module
from fastembed import SparseTextEmbedding
from search_core.models import SparseVector
from search_core.retrievals import UnifiedSparseAdapter
from sentence_transformers import SentenceTransformer


def create_mock_sparse_vector():
    return SparseVector(indices=[0, 1], values=[0.1, 0.2])


# ==========================================
# 1. FUNCTIONAL FACTORY & EXECUTION TESTS
# ==========================================


def test_from_sparse_text_embedding_and_encode():
    """Verify factory instantiation, tracking execution, and exact payload assertion."""
    mock_model = create_autospec(SparseTextEmbedding, instance=True)

    # Define our precise expected output payload
    expected_vectors = [create_mock_sparse_vector(), create_mock_sparse_vector()]
    mock_model.embed.return_value = expected_vectors

    # if hasattr(mock_model, "encode"):
    #     del mock_model.encode

    with patch.object(target_module, "SparseTextEmbedding") as mock_class:
        mock_class.return_value = mock_model

        adapter = UnifiedSparseAdapter.from_sparse_text_embedding("test-sparse-model", foo="bar")
        mock_class.assert_called_once_with(model_name="test-sparse-model", foo="bar")

        input_texts = ["hello", "world"]
        results = adapter.encode(input_texts)

        # Explicitly verify actual result matches the expected result
        assert results == expected_vectors
        mock_model.embed.assert_called_once_with(input_texts)


def test_from_sentence_transformer_and_encode_fallback():
    """Verify factory instantiation and fallback execution against exact array content match."""
    mock_model = create_autospec(SentenceTransformer, instance=True)

    expected_vectors = [create_mock_sparse_vector(), create_mock_sparse_vector()]
    mock_model.encode.return_value = expected_vectors

    # if hasattr(mock_model, "encode_sparse"):
    #     del mock_model.encode_sparse

    with patch.object(target_module, "SentenceTransformer") as mock_class:
        mock_class.return_value = mock_model

        adapter = UnifiedSparseAdapter.from_sentence_transformer("test-dense-model", device="cuda")
        mock_class.assert_called_once_with("test-dense-model", device="cuda")

        input_texts = ["query alpha", "query beta"]
        results = adapter.encode(input_texts)

        # Explicitly verify actual result matches the expected result
        assert results == expected_vectors
        mock_model.encode.assert_called_once_with(input_texts)


# ==========================================
# 2. ADVANCED BRANCH COVERAGE TESTS
# ==========================================


def test_encode_case_a_unified_model():
    """Case A: Hybrid model matching expected sparse output parsing logic."""
    mock_model = create_autospec(SentenceTransformer, instance=True)

    expected_vectors = [create_mock_sparse_vector(), create_mock_sparse_vector()]
    mock_model.encode.return_value = {"lexical_weights": expected_vectors}

    mock_model.encode.__code__ = MagicMock()
    mock_model.encode.__code__.co_varnames = ("texts", "return_dense", "return_sparse")

    adapter = UnifiedSparseAdapter(mock_model)
    input_texts = ["text1", "text2"]
    results = adapter.encode(input_texts)

    # Explicitly verify actual result matches the expected result unpacked from the dict
    assert results == expected_vectors
    mock_model.encode.assert_called_once_with(input_texts, return_dense=False, return_sparse=True)


def test_encode_case_c1_dedicated_sparse_method():
    """Case C1: Model utilizes a non-standard specialized .encode_sparse() method."""
    mock_model = create_autospec(SentenceTransformer, instance=True)

    expected_vectors = [create_mock_sparse_vector(), create_mock_sparse_vector()]
    mock_model.encode_sparse = MagicMock()
    mock_model.encode_sparse.return_value = expected_vectors

    adapter = UnifiedSparseAdapter(mock_model)
    input_texts = ["text1", "text2"]
    results = adapter.encode(input_texts)

    # Explicitly verify actual result matches the expected result from encode_sparse
    assert results == expected_vectors
    mock_model.encode_sparse.assert_called_once_with(input_texts)


def test_encode_case_c2_dense_matrix_error():
    """Case C2 Error: Catches real multi-dimensional dense arrays returning incorrectly from standard models."""
    mock_model = create_autospec(SentenceTransformer, instance=True)
    mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)

    # if hasattr(mock_model, "encode_sparse"):
    #     del mock_model.encode_sparse

    adapter = UnifiedSparseAdapter(mock_model)

    with pytest.raises(TypeError) as exc_info:
        adapter.encode(["text1", "text2"])

    assert "returned a dense matrix" in str(exc_info.value)


def test_encode_missing_methods_error():
    """Validates structural fallback error logic when no recognizable vector methods exist."""

    class CompletelyInvalidModel:
        pass

    adapter = UnifiedSparseAdapter(CompletelyInvalidModel())

    with pytest.raises(AttributeError) as exc_info:
        adapter.encode(["text1", "text2"])

    assert "does not have a valid embed or encode method" in str(exc_info.value)
