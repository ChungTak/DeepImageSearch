# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
SearchEngine — high-level unified API for DeepImageSearch v3.

Usage:
    from DeepImageSearch import SearchEngine

    engine = SearchEngine(model_name="clip-vit-b-32")

    # Index from folder path or list of paths
    engine.index("./photos")
    engine.index(["img1.jpg", "img2.jpg"])

    # Text search
    results = engine.search("a red car parked near a lake")

    # Image search
    results = engine.search("query.jpg")

    # Hybrid search
    results = engine.search("outdoor scene", image_query="photo.jpg")

    # With LLM captioning (any OpenAI SDK-compatible provider)
    engine = SearchEngine(
        model_name="clip-vit-b-32",
        captioner_model="your-model-name",
        captioner_api_key="your-api-key",
        captioner_base_url="https://your-provider.com/v1",
    )
    engine.index("./photos", generate_captions=True)
"""

import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from PIL import Image

from DeepImageSearch.core.embeddings import EmbeddingManager, BaseEmbedding
from DeepImageSearch.core.indexer import Indexer
from DeepImageSearch.core.searcher import Searcher
from DeepImageSearch.core.captioner import Captioner
from DeepImageSearch.vectorstores.base import BaseVectorStore
from DeepImageSearch.vectorstores.faiss_store import FAISSStore
from DeepImageSearch.metadatastore.base import BaseMetadataStore
from DeepImageSearch.metadatastore.json_store import JsonMetadataStore

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    All-in-one image search engine with multimodal support.

    Parameters
    ----------
    model_name : str
        Embedding model name. Use CLIP presets for text+image search:
        'clip-vit-b-32', 'clip-vit-l-14', 'siglip-vit-b-16', etc.
        Or timm model names for image-only: 'vgg19', 'resnet50', etc.
    vector_store : str or BaseVectorStore
        'faiss' (default), 'chroma', 'qdrant', or a custom BaseVectorStore instance.
    metadata_store : BaseMetadataStore or None
        Image record store. None creates a default JsonMetadataStore.
    index_dir : str
        Directory to persist the index and image records.
    device : str or None
        'cuda', 'mps', 'cpu', or None for auto-detect.
    index_type : str
        FAISS index type: 'flat', 'ivf', 'hnsw' (only for faiss backend).
    batch_size : int
        Batch size for embedding extraction.
    captioner_model : str or None
        Vision LLM model name. None to disable captioning.
        Uses OpenAI SDK standard — works with any compatible provider.
    captioner_api_key : str or None
        API key for the captioning provider.
    captioner_base_url : str or None
        API base URL for the captioning provider.
    chroma_collection : str
        ChromaDB collection name (only for chroma backend).
    qdrant_location : str or None
        Qdrant server URL (only for qdrant backend).
    """

    def __init__(
        self,
        model_name: str = "clip-vit-b-32",
        vector_store: Union[str, BaseVectorStore] = "faiss",
        metadata_store: Optional[BaseMetadataStore] = None,
        index_dir: str = "deep_image_search_index",
        device: Optional[str] = None,
        index_type: str = "flat",
        batch_size: int = 64,
        captioner_model: Optional[str] = None,
        captioner_api_key: Optional[str] = None,
        captioner_base_url: Optional[str] = None,
        chroma_collection: str = "deep_image_search",
        qdrant_location: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.index_dir = index_dir
        self.batch_size = batch_size

        # Create embedding backend
        self.embedding: BaseEmbedding = EmbeddingManager.create(
            model_name, device=device, batch_size=batch_size, **kwargs
        )

        # Create or use provided vector store
        if isinstance(vector_store, BaseVectorStore):
            self.vector_store = vector_store
        else:
            self.vector_store = self._create_store(
                vector_store, index_dir, index_type, chroma_collection, qdrant_location
            )

        # Create metadata store (default: local JSON)
        if metadata_store is not None:
            self.metadata_store = metadata_store
        else:
            self.metadata_store = JsonMetadataStore()
            # Try loading existing records
            self.metadata_store.load(index_dir)

        # Create captioner if model + api_key + base_url are all provided
        self.captioner: Optional[Captioner] = None
        if captioner_model and captioner_api_key and captioner_base_url:
            self.captioner = Captioner(
                model=captioner_model,
                api_key=captioner_api_key,
                base_url=captioner_base_url,
            )

        # Create indexer and searcher
        self.indexer = Indexer(
            embedding=self.embedding,
            vector_store=self.vector_store,
            metadata_store=self.metadata_store,
            captioner=self.captioner,
            batch_size=batch_size,
        )
        self.searcher = Searcher(
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        logger.info(
            f"SearchEngine ready: model={model_name}, "
            f"text_search={'yes' if self.embedding.supports_text else 'no'}, "
            f"dim={self.embedding.dimension}"
        )

    def _create_store(
        self,
        store_type: str,
        index_dir: str,
        index_type: str,
        chroma_collection: str,
        qdrant_location: Optional[str],
    ) -> BaseVectorStore:
        if store_type == "faiss":
            store = FAISSStore(dimension=self.embedding.dimension, index_type=index_type)
            if os.path.exists(os.path.join(index_dir, "index.faiss")):
                store.load(index_dir)
                logger.info(f"Loaded existing FAISS index from {index_dir}")
            return store
        elif store_type == "chroma":
            from DeepImageSearch.vectorstores.chroma_store import ChromaStore
            return ChromaStore(collection_name=chroma_collection, persist_directory=index_dir)
        elif store_type == "qdrant":
            from DeepImageSearch.vectorstores.qdrant_store import QdrantStore
            return QdrantStore(
                path=index_dir,
                dimension=self.embedding.dimension,
                location=qdrant_location,
            )
        raise ValueError(f"Unknown vector_store: {store_type}. Use 'faiss', 'chroma', or 'qdrant'.")

    # ── Input normalization ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_image_paths(image_paths: Union[str, List[str]]) -> List[str]:
        """Normalize input to a list of image file paths."""
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                from DeepImageSearch.data.loader import Load_Data
                return Load_Data().from_folder([image_paths])
            else:
                return [image_paths]
        return image_paths

    # ── Indexing ────────────────────────────────────────────────────────────

    def index(
        self,
        image_paths: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        generate_captions: bool = False,
        caption_prompt: Optional[str] = None,
        save: bool = True,
    ) -> int:
        """
        Index images for search.

        Parameters
        ----------
        image_paths : str or list[str]
            Folder path, single image path, or list of image paths.
        metadata : list[dict] or None
            Extra metadata per image.
        generate_captions : bool
            Auto-generate captions using the configured LLM.
        caption_prompt : str or None
            Custom prompt for captioning.
        save : bool
            Persist the index to disk after indexing.

        Returns
        -------
        int
            Number of images indexed.
        """
        resolved = self._resolve_image_paths(image_paths)
        count = self.indexer.index(
            resolved,
            extra_metadata=metadata,
            generate_captions=generate_captions,
            caption_prompt=caption_prompt,
        )
        if save:
            self.save()
        return count

    def add_images(
        self,
        image_paths: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        generate_captions: bool = False,
        save: bool = True,
    ) -> int:
        """Add new images to an existing index."""
        resolved = self._resolve_image_paths(image_paths)
        count = self.indexer.add_images(resolved, extra_metadata=metadata, generate_captions=generate_captions)
        if save:
            self.save()
        return count

    # ── Search ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: Union[str, Image.Image, np.ndarray],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        mode: Literal["auto", "text", "image", "hybrid", "interleaved"] = "auto",
        text_weight: float = 0.5,
        image_query: Optional[Union[str, Image.Image]] = None,
        interleaved: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for images.

        Parameters
        ----------
        query : str, PIL.Image, or np.ndarray
            Text query, image, or pre-computed vector.
        k : int
            Number of results.
        filters : dict or None
            Metadata filters.
        mode : str
            'auto', 'text', 'image', 'hybrid', or 'interleaved'.
        text_weight : float
            In hybrid mode, weight for text vs image (0-1).
        image_query : str or PIL.Image or None
            Secondary image for hybrid mode.
        interleaved : bool
            Whether to use interleaved single-pass multimodal encoding (EmbeddingGemma 2).

        Returns
        -------
        list[dict]
            Results with 'id', 'score', 'metadata' (includes image_id,
            image_index, image_name, image_path, caption, indexed_at).
        """
        return self.searcher.search(
            query, k=k, filters=filters, mode=mode,
            text_weight=text_weight, image_query=image_query,
            interleaved=interleaved,
        )

    def search_by_text(self, text: str, k: int = 10, filters: Optional[Dict[str, Any]] = None):
        """Text-to-image search (requires CLIP model)."""
        return self.searcher.search_by_text(text, k=k, filters=filters)

    def search_by_image(self, image_path: str, k: int = 10, filters: Optional[Dict[str, Any]] = None):
        """Image-to-image search."""
        return self.searcher.search_by_image(image_path, k=k, filters=filters)

    # ── Backward-compatible methods ─────────────────────────────────────────

    def get_similar_images(self, image_path: str, number_of_images: int = 10) -> Dict[int, str]:
        """v2-compatible: returns {index: image_path} dict."""
        return self.searcher.get_similar_images(image_path, number_of_images)

    def plot_similar_images(self, image_path: str, number_of_images: int = 6) -> None:
        """Plot query image and similar results using matplotlib."""
        self.searcher.plot_similar_images(image_path, number_of_images)

    # ── Records ─────────────────────────────────────────────────────────────

    def get_records(self) -> List[Dict[str, Any]]:
        """Return all image records as a list of dicts."""
        return [r.to_dict() for r in self.metadata_store.list_all()]

    def get_record(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Look up a single image record by ID."""
        record = self.metadata_store.get(image_id)
        return record.to_dict() if record else None

    # ── Deletion ────────────────────────────────────────────────────────────

    def delete(self, image_ids: Union[str, List[str]], save: bool = True) -> int:
        """
        Delete images from vector index and metadata store by image IDs.

        Parameters
        ----------
        image_ids : str or list[str]
            Single image ID or list of image IDs to delete.
        save : bool
            Whether to save index and metadata to disk after deletion.

        Returns
        -------
        int
            Number of records removed.
        """
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        if not image_ids:
            return 0

        before_count = self.vector_store.count()
        self.vector_store.delete(image_ids)
        self.metadata_store.delete(image_ids)
        after_count = self.vector_store.count()
        deleted = max(0, before_count - after_count)

        if save:
            self.save()
        return deleted

    def clear(self, save: bool = True) -> int:
        """
        Clear all indexed vectors and metadata records.

        Parameters
        ----------
        save : bool
            Whether to save the emptied index to disk.

        Returns
        -------
        int
            Number of cleared records.
        """
        before_count = self.vector_store.count()
        all_ids = []
        if hasattr(self.vector_store, "_ids") and self.vector_store._ids:
            all_ids = list(self.vector_store._ids)
        elif hasattr(self.metadata_store, "list_all"):
            all_ids = [r.image_id for r in self.metadata_store.list_all()]

        if all_ids:
            self.vector_store.delete(all_ids)
            self.metadata_store.delete(all_ids)
        else:
            if hasattr(self.vector_store, "_build_index") and hasattr(self.vector_store, "dimension"):
                self.vector_store._build_index(self.vector_store.dimension)
                self.vector_store._ids = []
                self.vector_store._metadata = []
            if hasattr(self.metadata_store, "_records"):
                self.metadata_store._records.clear()

        if save:
            self.save()
        return before_count

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, index_dir: Optional[str] = None) -> None:
        """Save the vector index and image records to disk."""
        target_dir = index_dir or self.index_dir
        self.vector_store.save(target_dir)
        self.metadata_store.save(target_dir)
        logger.info(f"Index saved to {target_dir}")

    def load(self, index_dir: Optional[str] = None) -> None:
        """Load an existing index and records from disk."""
        target_dir = index_dir or self.index_dir
        self.vector_store.load(target_dir)
        self.metadata_store.load(target_dir)
        logger.info(f"Index loaded from {target_dir} ({self.vector_store.count()} vectors)")

    # ── Info ────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of indexed images."""
        return self.vector_store.count()

    @property
    def supports_text_search(self) -> bool:
        """Whether text-to-image search is available."""
        return self.embedding.supports_text

    def info(self) -> Dict[str, Any]:
        """Return summary info about the search engine."""
        return {
            "model": self.model_name,
            "dimension": self.embedding.dimension,
            "indexed_images": self.count,
            "supports_text_search": self.supports_text_search,
            "vector_store": type(self.vector_store).__name__,
            "metadata_store": type(self.metadata_store).__name__,
            "index_dir": self.index_dir,
        }

    def __repr__(self) -> str:
        return (
            f"SearchEngine(model='{self.model_name}', "
            f"images={self.count}, "
            f"text_search={self.supports_text_search})"
        )
