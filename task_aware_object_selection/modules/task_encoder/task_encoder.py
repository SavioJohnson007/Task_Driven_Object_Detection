from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "The 'sentence-transformers' package is required for task encoding. "
        "Install it before using the task encoder."
    ) from exc


DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"

TASK_DESCRIPTIONS = [
    "Step on something to reach top of a shelf",
    "Sit comfortably",
    "Place flowers",
    "Get potatoes out of fire",
    "Water plant",
    "Get lemon out of tea",
    "Dig hole",
    "Open bottle of beer",
    "Open parcel",
    "Serve wine",
    "Pour sugar",
    "Smear butter",
    "Extinguish fire",
    "Pound carpet",
]


@dataclass(frozen=True)
class TaskMatch:
    task_id: int
    task_description: str
    similarity: float
    prompt_embedding: np.ndarray
    task_embeddings: np.ndarray
    best_task_embedding: np.ndarray


class TaskEncoder:
    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self._task_embeddings: np.ndarray | None = None

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        return embeddings

    def encode_prompt(self, prompt: str) -> np.ndarray:
        return self.encode([prompt])[0]

    def encode_tasks(self) -> np.ndarray:
        if self._task_embeddings is None:
            self._task_embeddings = self.encode(TASK_DESCRIPTIONS)
        return self._task_embeddings

    def match_prompt_to_task(self, prompt: str) -> TaskMatch:
        prompt_embedding = self.encode_prompt(prompt)
        task_embeddings = self.encode_tasks()
        similarities = cosine_similarities(prompt_embedding, task_embeddings)
        best_index = int(np.argmax(similarities))
        return TaskMatch(
            task_id=best_index + 1,
            task_description=TASK_DESCRIPTIONS[best_index],
            similarity=float(similarities[best_index]),
            prompt_embedding=prompt_embedding,
            task_embeddings=task_embeddings,
            best_task_embedding=task_embeddings[best_index],
        )


def cosine_similarities(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    query = query.astype(np.float32)
    corpus = corpus.astype(np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    corpus_norm = np.linalg.norm(corpus, axis=1, keepdims=True)
    dot_products = np.dot(query, corpus.T)
    similarity = dot_products / (query_norm * corpus_norm.T + 1e-10)
    return similarity[0]


def create_default_task_encoder(model_name: str | None = None, device: str | None = None) -> TaskEncoder:
    return TaskEncoder(model_name=model_name or DEFAULT_TEXT_MODEL, device=device)


def get_task_description_by_id(task_id: int) -> str:
    if not 1 <= task_id <= len(TASK_DESCRIPTIONS):
        raise ValueError(f"Task ID must be between 1 and {len(TASK_DESCRIPTIONS)}")
    return TASK_DESCRIPTIONS[task_id - 1]


def get_task_embeddings() -> np.ndarray:
    encoder = create_default_task_encoder()
    return encoder.encode_tasks()
