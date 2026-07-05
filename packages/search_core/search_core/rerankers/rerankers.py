from dataclasses import replace
from typing import Sequence

from search_core.models.models import SearchResponse


class Reranker:
    def __init__(self, model):
        self.model = model

    def rerank(
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
                pairs,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )

            idx = 0
            batch_result = []
            for doc_list in batch:
                num_matches = len(doc_list.matches)
                cur_scores = scores[idx : idx + num_matches]
                idx += num_matches

                # 1. Create cloned SearchResult objects containing the new rerank_score
                updated_matches = [
                    replace(doc, rerank_score=float(score))
                    for doc, score in zip(doc_list.matches, cur_scores)
                ]

                # 2. Sort the updated matches by the new rerank_score
                sorted_matches = sorted(updated_matches, key=lambda x: x.rerank_score, reverse=True)

                # 3. Clone the SearchResponse container with the new sorted matches list
                reranked_response = replace(doc_list, matches=sorted_matches)

                batch_result.append(reranked_response)

            yield batch_result
