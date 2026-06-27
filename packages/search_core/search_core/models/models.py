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


@dataclass(frozen=True)
class Document:
    """Raw document entering the indexing system."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
        # 1. Safely check for None without triggering Numpy array evaluation
        if self.embedding is None:
            raise ValueError(f"Missing embedding for document ID: {self.id}")

        # 2. Validate dimensional boundaries
        if len(self.embedding) != vector_size:
            raise ValueError(
                f"Embedding size mismatch for document {self.id}: "
                f"Expected {vector_size} dimensions, got {len(self.embedding)}."
            )

        # 3. Validate id format (Qdrant allows unsigned int or UUID)
        is_valid_uuid = False
        try:
            uuid.UUID(self.id)
            is_valid_uuid = True
        except ValueError:
            pass  # It's not a UUID, we'll check if it's an integer string next

        is_valid_uint = self.id.isdigit()

        if not (is_valid_uuid or is_valid_uint):
            raise ValueError(
                f"Invalid document ID: '{self.id}'. Qdrant requires document IDs to be "
                "either an unsigned integer or a valid UUID string."
            ) from None


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
