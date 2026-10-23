# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Thread-safe engine state manager for DeepImageSearch HTTP API Server."""

import hashlib
import inspect
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from DeepImageSearch import SearchEngine
from DeepImageSearch.server.config import ServerConfig

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


class EngineManager:
    """Thread-safe wrapper around SearchEngine for HTTP Server operations."""

    def __init__(self, config: ServerConfig, engine: Optional[Any] = None):
        self.config = config
        self._lock = threading.RLock()
        os.makedirs(self.config.upload_dir, exist_ok=True)
        os.makedirs(self.config.index_path, exist_ok=True)
        self.engine = engine if engine is not None else self._init_engine()

    def _init_engine(self) -> Any:
        init_sig = inspect.signature(SearchEngine.__init__)
        kwargs: Dict[str, Any] = {"model_name": self.config.model_name}

        if "vector_store" in init_sig.parameters:
            kwargs["vector_store"] = self.config.vector_store
        if "device" in init_sig.parameters and self.config.device:
            kwargs["device"] = self.config.device
        if "index_dir" in init_sig.parameters:
            kwargs["index_dir"] = self.config.index_path
        elif "index_path" in init_sig.parameters:
            kwargs["index_path"] = self.config.index_path

        if "captioner_model" in init_sig.parameters and self.config.captioner_model:
            kwargs["captioner_model"] = self.config.captioner_model
            kwargs["captioner_api_key"] = self.config.captioner_api_key
            kwargs["captioner_base_url"] = self.config.captioner_base_url

        engine = SearchEngine(**kwargs)
        self._try_autoload(engine)
        return engine

    def _try_autoload(self, engine: Any) -> None:
        if not os.path.exists(self.config.index_path):
            return
        faiss_file = os.path.join(self.config.index_path, "index.faiss")
        if not os.path.exists(faiss_file):
            return
        try:
            if hasattr(engine, "load"):
                sig = inspect.signature(engine.load)
                if len(sig.parameters) > 0:
                    engine.load(self.config.index_path)
                else:
                    engine.load()
            elif hasattr(engine, "vector_store"):
                engine.vector_store.load(self.config.index_path)
                if hasattr(engine, "metadata_store") and hasattr(
                    engine.metadata_store, "load"
                ):
                    engine.metadata_store.load(self.config.index_path)
        except Exception as exc:
            logger.warning("Index autoload skipped or empty at %s: %s", self.config.index_path, exc)

    @property
    def count(self) -> int:
        if hasattr(self.engine, "count"):
            val = self.engine.count
            return val() if callable(val) else int(val)
        if hasattr(self.engine, "vector_store"):
            return int(self.engine.vector_store.count())
        return 0

    def info(self) -> Dict[str, Any]:
        if hasattr(self.engine, "info") and callable(self.engine.info):
            data = self.engine.info()
            if isinstance(data, dict):
                return data
        return {
            "model_name": self.config.model_name,
            "vector_store": self.config.vector_store,
            "indexed_count": self.count,
            "index_path": self.config.index_path,
        }

    def save(self, target_path: Optional[str] = None) -> str:
        path = target_path or self.config.index_path
        with self._lock:
            os.makedirs(path, exist_ok=True)
            if hasattr(self.engine, "index_dir"):
                self.engine.index_dir = path
            if hasattr(self.engine, "save"):
                sig = inspect.signature(self.engine.save)
                if len(sig.parameters) > 0:
                    self.engine.save(path)
                else:
                    self.engine.save()
            else:
                if hasattr(self.engine, "vector_store") and hasattr(
                    self.engine.vector_store, "save"
                ):
                    self.engine.vector_store.save(path)
                if hasattr(self.engine, "metadata_store") and hasattr(
                    self.engine.metadata_store, "save"
                ):
                    self.engine.metadata_store.save(path)
        return path

    def load(self, source_path: Optional[str] = None) -> str:
        path = source_path or self.config.index_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index path does not exist: {path}")
        with self._lock:
            if hasattr(self.engine, "index_dir"):
                self.engine.index_dir = path
            if hasattr(self.engine, "load"):
                sig = inspect.signature(self.engine.load)
                if len(sig.parameters) > 0:
                    self.engine.load(path)
                else:
                    self.engine.load()
            else:
                if hasattr(self.engine, "vector_store") and hasattr(
                    self.engine.vector_store, "load"
                ):
                    self.engine.vector_store.load(path)
                if hasattr(self.engine, "metadata_store") and hasattr(
                    self.engine.metadata_store, "load"
                ):
                    self.engine.metadata_store.load(path)
        return path

    def index_paths(
        self,
        paths: Union[str, List[str]],
        generate_captions: bool = False,
        metadata: Optional[Any] = None,
        save_after_index: bool = True,
    ) -> int:
        with self._lock:
            before = self.count
            sig = inspect.signature(self.engine.index)
            kwargs: Dict[str, Any] = {}
            if "generate_captions" in sig.parameters:
                kwargs["generate_captions"] = generate_captions
            if "metadata" in sig.parameters and metadata is not None:
                kwargs["metadata"] = metadata
            if "save" in sig.parameters:
                kwargs["save"] = save_after_index

            indexed_return = self.engine.index(paths, **kwargs)
            after = self.count
            if save_after_index and "save" not in sig.parameters:
                self.save()

            if isinstance(indexed_return, int) and indexed_return > 0:
                return indexed_return
            return max(0, after - before)

    def save_uploaded_file(self, filename: str, content: bytes) -> str:
        ext = Path(filename or "image.jpg").suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{ext}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}"
            )
        digest = hashlib.sha256(content).hexdigest()[:16]
        safe_stem = "".join(
            c for c in Path(filename).stem if c.isalnum() or c in ("-", "_")
        )[:40] or "upload"
        dest_name = f"{safe_stem}_{digest}{ext}"
        dest_path = os.path.abspath(os.path.join(self.config.upload_dir, dest_name))
        with open(dest_path, "wb") as f:
            f.write(content)
        return dest_path

    def search(
        self,
        query: Optional[str] = None,
        image_query: Optional[str] = None,
        mode: Optional[str] = None,
        k: int = 10,
        text_weight: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        sig = inspect.signature(self.engine.search)
        kwargs: Dict[str, Any] = {}

        # Adapt top-k parameter name across SearchEngine versions
        if "k" in sig.parameters:
            kwargs["k"] = k
        elif "top_k" in sig.parameters:
            kwargs["top_k"] = k
        elif "number_of_images" in sig.parameters:
            kwargs["number_of_images"] = k

        if filters is not None and "filters" in sig.parameters:
            kwargs["filters"] = filters

        if mode == "hybrid" or (query and image_query):
            if "image_query" in sig.parameters:
                kwargs["image_query"] = image_query
            if "mode" in sig.parameters:
                kwargs["mode"] = mode or "hybrid"
            if "text_weight" in sig.parameters:
                kwargs["text_weight"] = text_weight
            elif "weight" in sig.parameters:
                kwargs["weight"] = text_weight
            elif "alpha" in sig.parameters:
                kwargs["alpha"] = text_weight
            raw_results = self.engine.search(query, **kwargs)
        elif image_query and not query:
            if "mode" in sig.parameters and mode:
                kwargs["mode"] = mode
            raw_results = self.engine.search(image_query, **kwargs)
        else:
            if "mode" in sig.parameters and mode:
                kwargs["mode"] = mode
            raw_results = self.engine.search(query, **kwargs)

        return self._normalize_results(raw_results)

    @staticmethod
    def _normalize_results(raw_results: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not raw_results:
            return normalized
        for item in raw_results:
            if isinstance(item, dict):
                normalized.append(
                    {
                        "id": str(item.get("id", item.get("image_id", ""))),
                        "score": float(item.get("score", 0.0)),
                        "metadata": dict(item.get("metadata", item)),
                    }
                )
            elif hasattr(item, "__dict__"):
                data = vars(item)
                normalized.append(
                    {
                        "id": str(data.get("id", data.get("image_id", ""))),
                        "score": float(data.get("score", 0.0)),
                        "metadata": dict(data.get("metadata", data)),
                    }
                )
        return normalized

    def get_records(self, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        all_records = self.engine.get_records() if hasattr(self.engine, "get_records") else []
        serialized = [
            r.to_dict() if hasattr(r, "to_dict") else (r if isinstance(r, dict) else vars(r))
            for r in all_records
        ]
        sliced = serialized[offset : offset + limit]
        return {
            "total": len(serialized),
            "offset": offset,
            "limit": limit,
            "records": sliced,
        }

    def get_record(self, image_id: str) -> Optional[Dict[str, Any]]:
        if not hasattr(self.engine, "get_record"):
            return None
        rec = self.engine.get_record(image_id)
        if rec is None:
            return None
        if hasattr(rec, "to_dict"):
            return rec.to_dict()
        if isinstance(rec, dict):
            return rec
        return vars(rec)
