import chromadb
from sentence_transformers import SentenceTransformer

from src import config
from src.notes import ParsedNote


def build_query(route: str, week_of) -> str:
    return f"Freight cost increase on the {route} route during the week of {week_of:%Y-%m-%d}"


def _document(note: ParsedNote) -> str:
    return f"{note.applies_to} ({note.note_date}): {note.text}"


class NoteIndex:
    def __init__(self, notes: list):
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        self._collection = chromadb.EphemeralClient().create_collection(
            "notes", metadata={"hnsw:space": "cosine"}, get_or_create=True
        )
        documents = [_document(n) for n in notes]
        self._collection.upsert(
            ids=[n.note_id for n in notes],
            documents=documents,
            embeddings=self._embed(documents),
        )

    def _embed(self, texts: list) -> list:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def search(self, query: str, k: int = config.TOP_K_NOTES) -> list:
        result = self._collection.query(
            query_embeddings=self._embed([query]),
            n_results=min(k, self._collection.count()),
        )
        return list(zip(result["ids"][0], result["distances"][0]))
