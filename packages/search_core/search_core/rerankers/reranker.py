from typing import Sequence

from search_core.models.models import SearchResponse


class Reranker:
    def __init__(self, model):
        self.model = model

    from typing import Sequence


def rerank_batch(
    self,
    retrieved_docs: Sequence[SearchResponse],
    batch_size: int = 32,
    show_progress_bar: bool = False,
):
    total = len(retrieved_docs)
    for i in range(0, total, batch_size):
        batch = retrieved_docs[i : i + batch_size]
        pairs = [(doc_list.query, doc.text) for doc_list in batch for doc in doc_list.matches]

        scores = self.model.predict(
            pairs, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_numpy=True
        )

        idx = 0
        batch_result = []
        for doc_list in batch:
            # 1. Slice the matching scores for this specific doc_list
            num_matches = len(doc_list.matches)
            cur_scores = scores[idx : idx + num_matches]
            idx += num_matches

            # 2. Pair documents with scores and sort them in descending order
            # (doc, score) -> sorted by score (x[1])
            sorted_docs_with_scores = sorted(
                zip(doc_list.matches, cur_scores), key=lambda x: x[1], reverse=True
            )

            # TODO: convert to a structured object
            batch_result.append((doc_list, sorted_docs_with_scores))

        yield batch_result
