from typing import Any, AsyncIterator, Iterator, Mapping, Protocol, Sequence
import numpy as np
from numpy.typing import NDArray

from search_core.models import (
    EmbeddedDocument,
    SearchConfig,
    EmbeddedQuery,
    SearchResponse,
    SparseVector,
)


class DenseEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[NDArray[np.float32]]:
        """Calculates dense embeddings for a batch of text strings.

        Args:
            texts: A sequence of raw text strings.

        Returns:
            A list of sparse vector representations.
        """
        ...


class SparseEncoder(Protocol):
    """Unified interface for sparse lexical embedding models."""

    def encode(self, texts: Sequence[str]) -> list[SparseVector]:
        """Calculates sparse embeddings for a batch of text strings.

        Args:
            texts: A sequence of raw text strings.

        Returns:
            A list of sparse vector representations.
        """
        ...


class VectorItem(Protocol):
    """Structural interface definition for all processing items."""

    id: str
    text: str
    embedding: NDArray[np.float32] | None
    sparse_vector: SparseVector | None


class VectorStore(Protocol):
    """Structural contract defining the requirements for a vector database driver."""

    def create_metadata_indexes(self, fields: list[str]) -> None:
        """Creates keyword payload indexes for accelerated field filtering.

        Args:
            fields: A list of string properties mapped within the metadata payload dictionary.
        """
        ...

    def save_embeddings(self, documents: list[EmbeddedDocument], wait: bool) -> int:
        """Translates Document data into points and pushes them to the backend API.

        Args:
            documents: Sequence of fully prepared documents holding raw texts, IDs, and vectors.
            wait: Boolean switch determining if the operation should block until safely written.

        Returns:
            The number of records submitted in the current execution loop.
        """
        ...

    def search(
        self,
        queries: Sequence[EmbeddedQuery],
        filters: Mapping[str, Any] | None,
        config: SearchConfig,
    ) -> Iterator[SearchResponse]:
        """Handles query routing and request orchestration against the backend.

        Args:
            queries: A batch list of inbound Query structures.
            filters: Filter keys mapped to primitive or sequence-based matching patterns.
            config: General constraints defining the batch mechanism and query boundaries.

        Returns:
            An iterator wrapping symmetric query result blocks.
        """
        ...


class AsyncVectorStore(Protocol):
    """Asynchronous structural contract defining the requirements for a vector database driver."""

    async def create_metadata_indexes(self, field_name: str) -> None:
        """Creates keyword payload indexes for accelerated field filtering asynchronously."""
        ...

    async def save_embeddings(
        self, documents: list[EmbeddedDocument], wait: bool
    ) -> int:
        """Translates Document data into points and pushes them to the backend API asynchronously."""
        ...

    async def search(
        self,
        queries: Sequence[EmbeddedQuery],
        filters: Mapping[str, Any] | None,
        config: SearchConfig,
    ) -> AsyncIterator[SearchResponse]:
        """Handles asynchronous query routing and request orchestration against the backend.

        Returns:
            An AsyncIterator wrapping symmetric query result blocks. Use `async for` to consume.
        """
        ...
