import pytest
from unittest.mock import ANY
from search_core.generators import GeneratorConfig, RagGenerator
from search_core.models import SearchResult

@pytest.fixture
def base_config():
    """Provides a standardized default configurations object for testing."""
    return GeneratorConfig(
        model_name="test-model",
        base_url="http://localhost:8000/v1",
        api_key="TEST_KEY",
        max_new_tokens=100,
        max_docs=2,
        temperature=0.7,
        top_p=0.8
    )


@pytest.fixture
def sample_docs():
    """Provides a collection of real SearchResult instances to test mapping and fallbacks."""
    doc1 = SearchResult(
        id="doc1",
        text="Content of document one.",
        score=0.9,
        metadata={"url": "https://example.com/1"}
    )
    doc2 = SearchResult(
        id="doc2",
        text="Content of document two.",
        score=0.8,
        metadata={"url": "https://example.com/2"}
    )
    doc3 = SearchResult(
        id="doc3",
        text="Content of document three.",
        score=0.7,
        metadata={}
    )
    return [doc1, doc2, doc3]


def test_init_initializes_openai_client(mocker, base_config):
    """Verify the OpenAI client initializes correctly with config variables."""
    # Patch the OpenAI class before initializing RagGenerator
    mock_openai_class = mocker.patch('search_core.generators.generators.OpenAI')
    
    generator = RagGenerator(base_config)
    
    mock_openai_class.assert_called_once_with(
        base_url=base_config.base_url, 
        api_key=base_config.api_key
    )
    assert generator.config == base_config


def test_answer_with_rag_empty_docs(mocker, base_config):
    """Verify fallback string is returned directly if docs list is empty."""
    mocker.patch('search_core.generators.generators.OpenAI')
    
    generator = RagGenerator(base_config)
    result = generator.answer_with_rag(query="What is testing?", docs=[])
    
    assert result == "I'm sorry, I don't have any source documents to answer that question."
    generator.client.chat.completions.create.assert_not_called()


def test_answer_with_rag_uses_config_defaults(mocker, base_config, sample_docs):
    """Verify API payload falls back to default config when no parameter overrides are provided."""
    # Mock the class
    mock_openai_class = mocker.patch('search_core.generators.generators.OpenAI')
    
    # Safely mock the deeply nested client instance call chain
    mock_create = mock_openai_class.return_value.chat.completions.create
    mock_create.return_value.choices = [mocker.MagicMock()]
    mock_create.return_value.choices[0].message.content = "  Mock response from LLM  "

    generator = RagGenerator(base_config)
    result = generator.answer_with_rag(query="Test Query", docs=sample_docs[:2])

    assert result == "Mock response from LLM"
    mock_create.assert_called_once_with(
        model="test-model",
        messages=ANY,
        max_tokens=100,
        temperature=0.7,
        top_p=0.8
    )


def test_answer_with_rag_parameter_overrides_and_temp_zero(mocker, base_config, sample_docs):
    """Verify overrides work, especially confirming that temperature=0.0 is preserved."""
    mock_openai_class = mocker.patch('search_core.generators.generators.OpenAI')
    mock_create = mock_openai_class.return_value.chat.completions.create
    mock_create.return_value.choices = [mocker.MagicMock()]
    mock_create.return_value.choices[0].message.content = "Response"

    generator = RagGenerator(base_config)
    generator.answer_with_rag(
        query="Test Query", 
        docs=sample_docs[:2],
        max_new_tokens=500,
        max_docs=1,
        temperature=0.0,
        top_p=0.95
    )

    mock_create.assert_called_once_with(
        model="test-model",
        messages=ANY,
        max_tokens=500,
        temperature=0.0,
        top_p=0.95
    )


def test_context_building_and_prompt_formatting(mocker, base_config, sample_docs):
    """Comprehensive verification of structural formatting, fallback ids, and prompt messages."""
    mock_openai_class = mocker.patch('search_core.generators.generators.OpenAI')
    mock_create = mock_openai_class.return_value.chat.completions.create
    mock_create.return_value.choices = [mocker.MagicMock()]
    mock_create.return_value.choices[0].message.content = "Response"

    generator = RagGenerator(base_config)
    target_docs = [sample_docs[0], sample_docs[2]]
    
    generator.answer_with_rag(query="What is the meaning of life?", docs=target_docs)

    # Extract the payload kwargs passed downstream to the mocked create instance method
    called_kwargs = mock_create.call_args[1]
    messages = called_kwargs['messages']

    expected_context = (
        f"--- Source https://example.com/1 ---\n{sample_docs[0]}\n"
        "\n"
        f"--- Source doc3 ---\n{sample_docs[2]}\n"
    )

    assert messages[0]["role"] == "system"
    assert "ONLY the provided text context" in messages[0]["content"]
    
    assert messages[1]["role"] == "user"
    assert f"=== START OF CONTEXT ===\n{expected_context}\n=== END OF CONTEXT ===" in messages[1]["content"]
    assert "Question: What is the meaning of life?" in messages[1]["content"]