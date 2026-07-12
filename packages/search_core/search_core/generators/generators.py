from dataclasses import dataclass

from openai import OpenAI

from search_core.models import SearchResult


@dataclass
class GeneratorConfig:
    model_name: str
    base_url: str
    api_key: str = "EMPTY"
    max_new_tokens: int = 1024
    max_docs: int = 10
    temperature: float = 0.6
    top_p: float = 0.9


class RagGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        # Use OpenAI client to connect to the vLLM instance
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def answer_with_rag(
        self,
        query: str,
        docs: list[SearchResult],
        max_new_tokens: int | None = None,
        max_docs: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        if not docs:
            return "I'm sorry, I don't have any source documents to answer that question."

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        max_docs = max_docs or self.config.max_docs
        # 'is not None' is required here because temperature=0.0 is a highly valid state
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p or self.config.top_p

        # Build context
        context_blocks = (
            f"--- Source {doc.metadata.get('url', doc.id) if doc.metadata else doc.id} ---\n{doc}\n"
            for doc in docs[:max_docs]
        )
        context = "\n".join(context_blocks)

        # Structured system instructions work significantly better
        # across open-weight models (Llama, Mistral, etc.)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise assistant. Your task is to answer the user's question "
                    "using ONLY the provided text context.\n\n"
                    "Strict Constraints:\n"
                    "1. Rely strictly on the clear facts mentioned in the context.\n"
                    "2. Do not assume or extrapolate outside the context.\n"
                    "3. If the context does not contain the answer, reply exactly with: "
                    "'I am sorry, but I do not have enough information in the provided sources to answer that.'"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Please answer the question based on the context below.\n\n"
                    f"=== START OF CONTEXT ===\n"
                    f"{context}\n"
                    f"=== END OF CONTEXT ===\n\n"
                    f"Question: {query}"
                ),
            },
        ]

        # Call the vLLM server
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            # No need to supply raw stop token IDs. vLLM naturally maps 'stop'
            # sequences to the current model's Chat template config.
        )

        return response.choices[0].message.content.strip()
