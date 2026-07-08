from unittest.mock import MagicMock, call

import pytest
from search_core.models import SearchResponse, SearchResult
from search_core.rerankers import Reranker


@pytest.fixture
def mock_model():
    """Provides a mocked cross-encoder model."""
    return MagicMock()


def test_rerank_single_batch_full_coverage(mock_model):
    """Verifies full object structure integrity, exact sorting order, and precise mock calls."""
    # Arrange
    doc1 = SearchResult(
        id="doc1", text="Python programming", score=0.4, metadata={"source": "wiki"}
    )
    doc2 = SearchResult(id="doc2", text="Java programming", score=0.9, metadata={"source": "git"})

    initial_response = SearchResponse(id="q1", query="What is python?", matches=[doc1, doc2])

    # Mocking predictable return scores. Let's make doc1 (Python) score higher after rerank
    mock_model.predict.return_value = [0.95, 0.15]

    reranker = Reranker(model=mock_model)

    # Act
    actual_batches = list(
        reranker.rerank([initial_response], batch_size=32, show_progress_bar=False)
    )

    # Assert - Full structural matching
    # We construct the *exact* expected object down to every single field to guarantee full coverage
    expected_response = SearchResponse(
        id="q1",
        query="What is python?",
        matches=[
            SearchResult(
                id="doc1",
                text="Python programming",
                score=0.4,
                rerank_score=0.95,
                metadata={"source": "wiki"},
            ),
            SearchResult(
                id="doc2",
                text="Java programming",
                score=0.9,
                rerank_score=0.15,
                metadata={"source": "git"},
            ),
        ],
    )

    assert actual_batches == [[expected_response]]

    # Assert - Model interactions
    mock_model.predict.assert_called_once_with(
        [("What is python?", "Python programming"), ("What is python?", "Java programming")],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def test_rerank_multiple_batches_full_coverage(mock_model):
    """Verifies multi-batch isolation, multi-call arguments, and comprehensive state validation."""
    # Arrange
    res1 = SearchResponse(
        id="q1", query="Q1", matches=[SearchResult(id="d1", text="T1", score=0.1)]
    )
    res2 = SearchResponse(
        id="q2", query="Q2", matches=[SearchResult(id="d2", text="T2", score=0.2)]
    )
    res3 = SearchResponse(
        id="q3", query="Q3", matches=[SearchResult(id="d3", text="T3", score=0.3)]
    )

    mock_model.predict.side_effect = [
        [0.88, 0.77],  # Scores for batch 1 (res1, res2)
        [0.99],  # Scores for batch 2 (res3)
    ]

    reranker = Reranker(model=mock_model)

    # Act
    actual_batches = list(reranker.rerank([res1, res2, res3], batch_size=2, show_progress_bar=True))

    # Expected structures
    expected_res1 = SearchResponse(
        id="q1",
        query="Q1",
        matches=[SearchResult(id="d1", text="T1", score=0.1, rerank_score=0.88)],
    )
    expected_res2 = SearchResponse(
        id="q2",
        query="Q2",
        matches=[SearchResult(id="d2", text="T2", score=0.2, rerank_score=0.77)],
    )
    expected_res3 = SearchResponse(
        id="q3",
        query="Q3",
        matches=[SearchResult(id="d3", text="T3", score=0.3, rerank_score=0.99)],
    )

    assert actual_batches == [[expected_res1, expected_res2], [expected_res3]]

    # Assert - Model Calls Sequence
    assert mock_model.predict.call_count == 2

    expected_calls = [
        call(
            [("Q1", "T1"), ("Q2", "T2")],
            batch_size=2,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        call([("Q3", "T3")], batch_size=2, show_progress_bar=True, convert_to_numpy=True),
    ]
    mock_model.predict.assert_has_calls(expected_calls, any_order=False)


def test_rerank_raises_value_error_when_no_model_provided():
    """
    Ensures a ValueError is raised if neither the class instance
    nor the method call is provided with a model.
    """
    # 1. Initialize without a model
    reranker = Reranker(model=None)

    # Mock data structure to get past the initial setup
    mock_doc = MagicMock(spec=SearchResponse)
    mock_doc.query = "test query"
    mock_doc.matches = [MagicMock()]

    # 3. Assert ValueError is raised when calling rerank without passing a model
    generator = reranker.rerank(retrieved_docs=[mock_doc], model=None)

    with pytest.raises(ValueError, match="A reranking model must be provided."):
        # We must trigger the generator execution using next()
        next(generator)


def test_rerank_raises_runtime_error_on_mismatched_scores_count():
    """
    Ensures a RuntimeError is raised if the model's prediction output length
    does not match the total number of query-document pairs.
    """
    # 1. Set up a mock model that returns an incorrect number of scores
    mock_model = MagicMock()
    # We will simulate 2 total query-document pairs, but the model returns only 1 score
    mock_model.predict.return_value = [0.95]

    reranker = Reranker()

    # 2. Create dummy search responses with a total of 2 matches
    mock_match1 = MagicMock()
    mock_match2 = MagicMock()

    doc_container = MagicMock(spec=SearchResponse)
    doc_container.query = "AI assistant"
    doc_container.matches = [mock_match1, mock_match2]  # 2 pairs total

    retrieved_docs = [doc_container]

    # 3. Assert RuntimeError is raised due to the length mismatch (1 score vs 2 pairs)
    generator = reranker.rerank(retrieved_docs=retrieved_docs, model=mock_model)

    with pytest.raises(RuntimeError, match="Model returned an unexpected number of scores."):
        next(generator)
