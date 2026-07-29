import os
import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from src.core.config import settings, get_embeddings
from src.core.logger import get_logger

logger = get_logger(__name__)


class ChromaClient:
    def __init__(self):
        self.persist_directory = settings.chroma_db_path
        self.bm25_index_path = settings.bm25_index_path
        self.embeddings = get_embeddings()
        self._vectorstore = None
        self._bm25_retriever = None

    # ------------------------------------------------------------------ #
    #  Vector store (Dense retrieval)
    # ------------------------------------------------------------------ #
    def connect(self):
        if not self._vectorstore:
            try:
                self._vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
                logger.info("Connected to ChromaDB successfully.")
            except Exception as e:
                logger.error(f"Failed to load ChromaDB: {e}")
                self._vectorstore = None

    # ------------------------------------------------------------------ #
    #  BM25 Sparse index persistence
    # ------------------------------------------------------------------ #
    def _save_bm25(self, bm25_retriever: BM25Retriever):
        """Persist the BM25 retriever to disk using pickle."""
        os.makedirs(os.path.dirname(self.bm25_index_path), exist_ok=True)
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        logger.info(f"BM25 index saved to {self.bm25_index_path}")

    def _load_bm25(self) -> BM25Retriever | None:
        """Load a previously persisted BM25 retriever from disk."""
        if os.path.exists(self.bm25_index_path):
            try:
                with open(self.bm25_index_path, "rb") as f:
                    retriever = pickle.load(f)
                logger.info("BM25 index loaded from disk.")
                return retriever
            except Exception as e:
                logger.warning(f"Could not load BM25 index: {e}")
        return None

    # ------------------------------------------------------------------ #
    #  Hybrid retriever (Dense + Sparse)
    # ------------------------------------------------------------------ #
    def get_retriever(self, k: int = 5, dense_weight: float = 0.5):
        """
        Returns an EnsembleRetriever combining:
          - Dense vector search via Chroma  (semantic)
          - Sparse keyword search via BM25   (exact match)
        Falls back to dense-only if no BM25 index is found.
        """
        self.connect()
        if not self._vectorstore:
            logger.error("ChromaDB is not available. Returning None.")
            return None

        dense_retriever = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3}
        )

        if not settings.use_bm25:
            logger.info("BM25 disabled by config. Returning dense-only retriever.")
            return dense_retriever

        # Try loading saved BM25 index
        if not self._bm25_retriever:
            self._bm25_retriever = self._load_bm25()

        if self._bm25_retriever:
            logger.info(f"Using HYBRID retriever (dense={dense_weight}, sparse={1-dense_weight})")
            return EnsembleRetriever(
                retrievers=[dense_retriever, self._bm25_retriever],
                weights=[dense_weight, 1.0 - dense_weight],
            )
        else:
            logger.warning("BM25 index not found. Falling back to dense-only retrieval.")
            return dense_retriever

    # ------------------------------------------------------------------ #
    #  Ingestion
    # ------------------------------------------------------------------ #
    def clear_database(self):
        """Delete ChromaDB directory and BM25 index on disk to start fresh."""
        import shutil
        self._vectorstore = None
        self._bm25_retriever = None
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info(f"Deleted ChromaDB directory at {self.persist_directory}")
            except Exception as e:
                logger.warning(f"Could not delete ChromaDB directory: {e}")
        if os.path.exists(self.bm25_index_path):
            try:
                os.remove(self.bm25_index_path)
                logger.info(f"Deleted BM25 index file at {self.bm25_index_path}")
            except Exception as e:
                logger.warning(f"Could not delete BM25 index file: {e}")

    def ingest_documents(self, documents):
        """
        Ingests document chunks into:
          1. ChromaDB (dense vector index)
          2. BM25 sparse index (persisted to disk)
        """
        try:
            logger.info(f"Ingesting {len(documents)} chunks into ChromaDB...")
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info("ChromaDB ingestion complete.")

            logger.info("Building BM25 sparse index...")
            self._bm25_retriever = BM25Retriever.from_documents(documents)
            self._save_bm25(self._bm25_retriever)

        except Exception as e:
            logger.error(f"Failed during ingestion: {e}")
            raise

