from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import Any, Iterator, Mapping, NamedTuple, Sequence

import numpy as np
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from search_core.interfaces import (
    DenseEncoder,
    SparseEncoder,
    VectorStore,
)
from search_core.models import (
    Document,
    SearchConfig,
    SearchMode,
    SearchQuery,
    SearchResponse,
    SparseVector,
    Vector,
)


class EmbeddingResult(NamedTuple):
    dense: Vector | None
    sparse: SparseVector | None


class UnifiedSparseAdapter:
    """Adapts various underlying models to conform to the SparseEncoder Protocol."""

    def __init__(self, shared_model: Any):
        self.model = shared_model

    @classmethod
    def from_sparse_text_embedding(cls, model_name: str, **kwargs) -> "UnifiedSparseAdapter":
        """Factory method to build the adapter directly from a SparseTextEmbedding model."""
        model_instance = SparseTextEmbedding(model_name=model_name, **kwargs)
        return cls(shared_model=model_instance)

    @classmethod
    def from_sentence_transformer(cls, model_name: str, **kwargs) -> "UnifiedSparseAdapter":
        """Factory method to build the adapter directly from a SentenceTransformer model."""
        model_instance = SentenceTransformer(model_name, **kwargs)
        return cls(shared_model=model_instance)

    def encode(self, texts: Sequence[str]) -> list[SparseVector]:
        # Case A: Unified/hybrid models that accept specific return flags
        if (
            hasattr(self.model, "encode")
            and "return_sparse" in self.model.encode.__code__.co_varnames
        ):
            outputs = self.model.encode(texts, return_dense=False, return_sparse=True)
            return list(outputs["lexical_weights"])

        # Case B: Standard FastEmbed sparse models (uses .embed)
        elif hasattr(self.model, "embed"):
            return list(self.model.embed(texts))

        # Case C: SentenceTransformer variants
        elif hasattr(self.model, "encode"):
            # Sub-case C1: Check if the model has a dedicated sparse encoding method
            if hasattr(self.model, "encode_sparse"):
                return list(self.model.encode_sparse(texts))

            # Sub-case C2: Fallback check to ensure we aren't getting back dense numpy arrays
            result = self.model.encode(texts)

            # If the result is a standard dense matrix (NumPy array), it's invalid for this adapter
            if isinstance(result, np.ndarray) and result.ndim == 2:
                raise TypeError(
                    f"The model '{type(self.model).__name__}' returned a dense matrix. "
                    "A standard SentenceTransformer cannot be used in a SparseAdapter "
                    "unless it is a dedicated sparse/hybrid architecture."
                )

            return list(result)

        else:
            raise AttributeError(
                f"The wrapped model {type(self.model).__name__} does not have a valid embed or encode method."
            )


class Retriever:
    def __init__(
        self,
        dense_model: DenseEncoder,
        sparse_model: SparseEncoder,
        embedding_store: VectorStore,
    ):
        """Initializes the orchestrator with its requisite dependencies.

        Args:
            dense_model: Inference client responsible for standard spatial encodings.
            sparse_model: Protocol-compliant wrapper managing lexical weight mapping.
            embedding_store: Data access instance managing database state.
        """
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.embedding_store = embedding_store

    def _create_embeddings(
        self,
        texts: list[str],
        dense: bool = True,
        sparse: bool = True,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        dense_embeddings, sparse_vectors = None, None

        if dense:
            dense_embeddings = self.dense_model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

        if sparse:
            sparse_vectors = self.sparse_model.encode(texts)

        return EmbeddingResult(dense=dense_embeddings, sparse=sparse_vectors)

    def create_metadata_indexes(self, fields: list[str]) -> None:
        """Proxies index construction commands down to the datastore integration layer.

        Args:
            fields: Identifiers for the fields to be indexed.
        """
        if not fields:
            raise ValueError("Cannot create index on empty fields.")

        # Deduplicate fields to prevent redundant thread executions
        unique_fields = list(set(fields))

        # Cap concurrency at a maximum of 8 workers
        max_workers = min(len(unique_fields), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Materialize list to ensure execution completes before exiting context
            list(executor.map(self.embedding_store.create_metadata_index, unique_fields))

    def create_and_save_embeddings(
        self,
        documents: list[Document],
        create_sparse_embeddings: bool = True,
        wait: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Iterator[int]:
        """Creates embeddings for documents and saves them in batches.

        Args:
            documents: Documents to embed and store.
            create_sparse_embeddings: Whether to generate sparse embeddings in addition
                to dense embeddings.
            wait: Whether to wait for the embedding store to persist each batch.
            batch_size: Number of documents processed per batch.
            show_progress: Whether to display embedding progress.

        Yields:
            The cumulative number of documents saved after each batch.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if not documents:
            raise ValueError("documents cannot be empty")

        total_saved = 0
        total_docs = len(documents)

        for i in range(0, total_docs, batch_size):
            docs = documents[i : i + batch_size]
            texts = [doc.text for doc in docs]
            embeddings = self._create_embeddings(
                texts=texts,
                dense=True,
                sparse=create_sparse_embeddings,
                show_progress=show_progress,
            )

            # If an embedding array is None, create an infinite iterator of None values
            # so zip() can gracefully unpack it for every document in the batch.
            dense_iter = embeddings.dense if embeddings.dense is not None else repeat(None)
            sparse_iter = embeddings.sparse if embeddings.sparse is not None else repeat(None)

            embedded_docs = [
                doc.to_embedded(embedding, sp_vector)
                for doc, embedding, sp_vector in zip(docs, dense_iter, sparse_iter)
            ]

            total_saved += self.embedding_store.save_embeddings(documents=embedded_docs, wait=wait)
            yield total_saved

    def retrieve_documents(
        self,
        queries: list[SearchQuery],
        filters: Mapping[str, Any] | None,
        config: SearchConfig,
        batch_size: int = 32,
    ) -> Iterator[SearchResponse]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        total_queries = len(queries)

        for i in range(0, total_queries, batch_size):
            batched_queries = queries[i : i + batch_size]
            texts = [query.text for query in batched_queries]
            embeddings = self._create_embeddings(
                texts=texts,
                dense=True,
                sparse=config.mode == SearchMode.HYBRID,
                show_progress=False,
            )

            # If an embedding array is None, create an infinite iterator of None values
            # so zip() can gracefully unpack it for every document in the batch.
            dense_iter = embeddings.dense if embeddings.dense is not None else repeat(None)
            sparse_iter = embeddings.sparse if embeddings.sparse is not None else repeat(None)

            embedded_queries = [
                query.to_embedded(embedding, sp_vector)
                for query, embedding, sp_vector in zip(batched_queries, dense_iter, sparse_iter)
            ]

            yield self.embedding_store.search(embedded_queries, filters, config)
