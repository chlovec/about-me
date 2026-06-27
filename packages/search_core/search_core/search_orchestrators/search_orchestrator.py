from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Sequence

from search_core.interfaces import (
    DenseEncoder,
    SparseEncoder,
    VectorItem,
    VectorStore,
)
from search_core.models import Document, EmbeddedDocument


class SearchOrchestrator:
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
        items: Sequence[VectorItem],
        dense: bool = True,
        sparse: bool = True,
        show_progress: bool = False,
    ):
        if not items:
            return

        texts = [item.text for item in items]

        if dense:
            embeddings = self.dense_model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            for item, embedding in zip(items, embeddings):
                item.embedding = embedding

        if sparse:
            sparse_vectors = self.sparse_model.encode(texts)
            for item, sp_vector in zip(items, sparse_vectors):
                item.sparse_vector = sp_vector

    def create_metadata_indexes(self, fields: list[str]) -> None:
        """Proxies index construction commands down to the datastore integration layer.

        Args:
            fields: Identifiers for the fields to be indexed.
        """
        if not fields:
            raise ValueError("Cannot create index on empty fields.")

        # Deduplicate fields to prevent redundant thread executions
        unique_fields = list(set(fields))

        # Cap concurrency at a maximum of 16 workers
        max_workers = min(len(unique_fields), 16)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

        total_saved = 0
        total_docs = len(documents)

        for i in range(0, total_docs, batch_size):
            docs = documents[i : i + batch_size]
            embedded_docs = [EmbeddedDocument.from_document(doc) for doc in docs]
            self._create_embeddings(
                items=embedded_docs,
                dense=True,
                sparse=create_sparse_embeddings,
                show_progress=show_progress,
            )
            total_saved += self.embedding_store.save_embeddings(documents=embedded_docs, wait=wait)
            yield total_saved
