import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

# -------------------------
# Core Types
# -------------------------
Vector = NDArray[np.float32]


class SearchMode(StrEnum):
    """Supported search modes for the vector store."""

    DENSE = "dense"
    HYBRID = "hybrid"
    SPARSE = "sparse"


@dataclass(frozen=True)
class SparseVector:
    """Sparse vector representation for lexical or hybrid retrieval."""

    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        """Automatically validates the vector upon instantiation."""
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"Sparse vector mismatch: {len(self.indices)} indices vs {len(self.values)} values."
            )


# -------------------------
# Input Contracts
# -------------------------


@dataclass(frozen=True)
class SearchQuery:
    """Public search request DTO."""

    id: str
    text: str

    def to_embedded(
        self, embedding: Vector, sparse_vector: SparseVector | None = None
    ) -> "EmbeddedQuery":
        return EmbeddedQuery(
            id=self.id,
            text=self.text,
            embedding=embedding,
            sparse_vector=sparse_vector,
        )


@dataclass(frozen=True)
class Document:
    """Raw document entering the indexing system."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_embedded(
        self, embedding: Vector, sparse_vector: SparseVector | None = None
    ) -> "EmbeddedDocument":
        """Converts the base document into an EmbeddedDocument with spatial vectors."""
        return EmbeddedDocument(
            id=self.id,
            text=self.text,
            metadata=self.metadata,
            embedding=embedding,
            sparse_vector=sparse_vector,
        )


# -------------------------
# Output Contracts
# -------------------------


@dataclass(frozen=True)
class SearchResult:
    """Single retrieved document with ranking score."""

    id: str
    text: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResponse:
    """Final response for a search query execution."""

    id: str
    matches: list[SearchResult]


# -------------------------
# Internal Processing Models
# -------------------------


@dataclass(frozen=True)
class EmbeddedQuery:
    """Query after feature extraction (dense + sparse)."""

    id: str
    text: str
    embedding: Vector | None
    sparse_vector: SparseVector | None = None


@dataclass(frozen=True)
class EmbeddedDocument:
    """Document enriched with vector representations."""

    id: str | int
    text: str
    embedding: Vector | None
    sparse_vector: SparseVector | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, vector_size: int) -> None:
        # Safely check for None without triggering Numpy array evaluation
        if self.embedding is None:
            raise ValueError(f"Missing embedding for document ID: {self.id}")

        # Validate dimensional boundaries
        if len(self.embedding) != vector_size:
            raise ValueError(
                f"Embedding size mismatch for document {self.id}: "
                f"Expected {vector_size} dimensions, got {len(self.embedding)}."
            )

        # Validate id type
        if not isinstance(self.id, str):
            raise ValueError(
                f"Invalid document ID type: {type(self.id).__name__}. "
                "Qdrant requires document IDs to be either an unsigned integer or "
                "a valid UUID string."
            )

        # (Qdrant allows unsigned int or UUID)
        # Validate unsigned int id format
        if isinstance(self.id, int):
            if self.id < 0:
                raise ValueError(
                    f"Invalid document ID: '{self.id}'. Qdrant requires document IDs to be "
                    "either an unsigned integer or a valid UUID string."
                )
            return

        # 4. Validate id format (Qdrant allows unsigned int or UUID)
        try:
            uuid.UUID(self.id)
        except ValueError as exc:
            raise ValueError(
                f"Invalid document ID: '{self.id}'. Qdrant requires document IDs to be "
                "either an unsigned integer or a valid UUID string."
            ) from exc


# -------------------------
# Configuration Classes
# -------------------------
@dataclass
class SearchConfig:
    """Execution parameters and constraints for vector store search operations."""

    # Target counts
    k: int = 10
    prefetch_k: int = 100
    rrf_k: int = 60

    # Batch processing configuration
    batch_size: int = 32

    # Search behavior strategy
    mode: SearchMode = SearchMode.HYBRID

    # Selection and projection
    return_metadata: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validates search configuration thresholds.

        Raises:
            ValueError: If any thresholds violate reciprocal rank fusion or limit logic.
        """
        if self.k <= 0:
            raise ValueError("k must be greater than 0")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
