from search_core.interfaces import DenseEncoder, EmbeddingStore, SparseEncoder


class SearchOrchestrator:
    def __init__(
        self,
        dense_model: DenseEncoder,
        sparse_model: SparseEncoder,
        embedding_store: EmbeddingStore,
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

    def _create_embeddings(self):
        pass
