# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Thread-safe engine state manager for DeepImageSearch HTTP API Server."""

import copy
import hashlib
import inspect
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from DeepImageSearch import SearchEngine
from DeepImageSearch.core.filter_utils import apply_night_color_tolerance
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
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        kwargs: Dict[str, Any] = {}

        # Adapt top-k parameter name across SearchEngine versions
        if "k" in sig.parameters or has_var_kw:
            kwargs["k"] = k
        elif "top_k" in sig.parameters:
            kwargs["top_k"] = k
        elif "number_of_images" in sig.parameters:
            kwargs["number_of_images"] = k

        if filters is not None and ("filters" in sig.parameters or has_var_kw):
            kwargs["filters"] = filters

        if mode == "hybrid" or (query and image_query):
            if "image_query" in sig.parameters or has_var_kw:
                kwargs["image_query"] = image_query
            if "mode" in sig.parameters or has_var_kw:
                kwargs["mode"] = mode or "hybrid"
            if "text_weight" in sig.parameters or has_var_kw:
                kwargs["text_weight"] = text_weight
            elif "weight" in sig.parameters:
                kwargs["weight"] = text_weight
            elif "alpha" in sig.parameters:
                kwargs["alpha"] = text_weight
            raw_results = self.engine.search(query, **kwargs)
        elif image_query and not query:
            if "mode" in sig.parameters or has_var_kw:
                kwargs["mode"] = mode
            raw_results = self.engine.search(image_query, **kwargs)
        else:
            if "mode" in sig.parameters or has_var_kw:
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

    def search_dual(
        self,
        query: Optional[str] = None,
        ship_query: Optional[str] = None,
        scene_query: Optional[str] = None,
        image_query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        w1: float = 0.7,
        w2: float = 0.3,
        night_color_tolerance: bool = True,
        k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Execute dual-branch search combining ship crop and scene retrieval.

        1. Priority structured filtering (with night color tolerance applied).
        2. Dispatched retrieval: ship branch (s_ship) + scene branch (s_scene).
        3. Alignment by group_id (or track_id).
        4. Weighted score fusion: score = w1 * s_ship + w2 * s_scene.
        5. Return Top-K fused group results.
        """
        if not query and not ship_query and not scene_query and not image_query:
            raise ValueError(
                "Provide at least one of 'query', 'ship_query', 'scene_query', or 'file'."
            )

        eff_filters = copy.deepcopy(filters) if filters else {}
        if night_color_tolerance and eff_filters:
            eff_filters = apply_night_color_tolerance(eff_filters) or {}
        final_filters = eff_filters if eff_filters else None

        ship_text = ship_query or query
        scene_text = scene_query or query

        ship_mode = "hybrid" if (ship_text and image_query) else ("text" if ship_text else "image")
        scene_mode = "hybrid" if (scene_text and image_query) else ("text" if scene_text else "image")

        fetch_k = max(k * 10, 100)

        # Inspect if dataset contains explicit crop_type tags
        has_crop_types = False
        vector_store = getattr(self.engine, "vector_store", None)
        metadata_list = getattr(vector_store, "_metadata", None)
        if metadata_list and isinstance(metadata_list, list):
            for m in metadata_list[:200]:
                if isinstance(m, dict) and "crop_type" in m:
                    has_crop_types = True
                    break

        if has_crop_types:
            ship_filter = {"crop_type": "ship_crop"}
            scene_filter = {"crop_type": {"$in": ["context_crop", "full_frame", "scene", "frame"]}}
            if final_filters:
                ship_branch_filters = copy.deepcopy(final_filters)
                ship_branch_filters.update(ship_filter)
                scene_branch_filters = copy.deepcopy(final_filters)
                scene_branch_filters.update(scene_filter)
            else:
                ship_branch_filters = ship_filter
                scene_branch_filters = scene_filter
        else:
            ship_branch_filters = final_filters
            scene_branch_filters = final_filters

        # Branch 1: Ship retrieval
        ship_results = self.search(
            query=ship_text,
            image_query=image_query,
            mode=ship_mode,
            k=fetch_k,
            filters=ship_branch_filters,
        )

        # Branch 2: Scene retrieval
        scene_results = self.search(
            query=scene_text,
            image_query=image_query,
            mode=scene_mode,
            k=fetch_k,
            filters=scene_branch_filters,
        )

        def _extract_group_id(item: Dict[str, Any]) -> str:
            meta = item.get("metadata", {})
            return str(
                meta.get("group_id")
                or meta.get("track_id")
                or item.get("id")
                or "group_unknown"
            )

        group_data: Dict[str, Dict[str, Any]] = {}

        for res in ship_results:
            gid = _extract_group_id(res)
            entry = group_data.setdefault(
                gid,
                {
                    "group_id": gid,
                    "s_ship": 0.0,
                    "s_scene": 0.0,
                    "ship_crop_image_id": None,
                    "context_crop_image_id": None,
                    "full_frame_image_id": None,
                    "metadata": {},
                },
            )
            score = float(res.get("score", 0.0))
            if score > entry["s_ship"]:
                entry["s_ship"] = score

            meta = res.get("metadata", {})
            entry["metadata"].update(meta)

            crop_type = meta.get("crop_type")
            item_id = res.get("id")
            if crop_type == "ship_crop" and not entry["ship_crop_image_id"]:
                entry["ship_crop_image_id"] = item_id
            elif crop_type in ("context_crop", "context") and not entry["context_crop_image_id"]:
                entry["context_crop_image_id"] = item_id
            elif crop_type in ("full_frame", "scene", "frame") and not entry["full_frame_image_id"]:
                entry["full_frame_image_id"] = item_id
            elif not entry["ship_crop_image_id"]:
                entry["ship_crop_image_id"] = item_id

            if meta.get("ship_crop_image_id"):
                entry["ship_crop_image_id"] = meta["ship_crop_image_id"]
            if meta.get("context_crop_image_id"):
                entry["context_crop_image_id"] = meta["context_crop_image_id"]
            if meta.get("full_frame_image_id"):
                entry["full_frame_image_id"] = meta["full_frame_image_id"]

        for res in scene_results:
            gid = _extract_group_id(res)
            entry = group_data.setdefault(
                gid,
                {
                    "group_id": gid,
                    "s_ship": 0.0,
                    "s_scene": 0.0,
                    "ship_crop_image_id": None,
                    "context_crop_image_id": None,
                    "full_frame_image_id": None,
                    "metadata": {},
                },
            )
            score = float(res.get("score", 0.0))
            if score > entry["s_scene"]:
                entry["s_scene"] = score

            meta = res.get("metadata", {})
            entry["metadata"].update(meta)

            crop_type = meta.get("crop_type")
            item_id = res.get("id")
            if crop_type in ("context_crop", "context") and not entry["context_crop_image_id"]:
                entry["context_crop_image_id"] = item_id
            elif crop_type in ("full_frame", "scene", "frame") and not entry["full_frame_image_id"]:
                entry["full_frame_image_id"] = item_id
            elif crop_type == "ship_crop" and not entry["ship_crop_image_id"]:
                entry["ship_crop_image_id"] = item_id
            elif not entry["full_frame_image_id"]:
                entry["full_frame_image_id"] = item_id

            if meta.get("ship_crop_image_id"):
                entry["ship_crop_image_id"] = meta["ship_crop_image_id"]
            if meta.get("context_crop_image_id"):
                entry["context_crop_image_id"] = meta["context_crop_image_id"]
            if meta.get("full_frame_image_id"):
                entry["full_frame_image_id"] = meta["full_frame_image_id"]

        # Backfill missing crop image ids from metadata store if available
        if metadata_list and isinstance(metadata_list, list):
            ids_list = getattr(vector_store, "_ids", [])
            for gid, entry in group_data.items():
                if not (entry["ship_crop_image_id"] and entry["full_frame_image_id"]):
                    for idx, m in enumerate(metadata_list):
                        if isinstance(m, dict) and (m.get("group_id") == gid or m.get("track_id") == gid):
                            ctype = m.get("crop_type")
                            cid = ids_list[idx] if idx < len(ids_list) else None
                            if ctype == "ship_crop" and not entry["ship_crop_image_id"]:
                                entry["ship_crop_image_id"] = cid
                            elif ctype in ("context_crop", "context") and not entry["context_crop_image_id"]:
                                entry["context_crop_image_id"] = cid
                            elif ctype in ("full_frame", "scene", "frame") and not entry["full_frame_image_id"]:
                                entry["full_frame_image_id"] = cid

        fused_results = []
        for gid, entry in group_data.items():
            s_ship = float(entry["s_ship"])
            s_scene = float(entry["s_scene"])
            fused_score = round(float(w1 * s_ship + w2 * s_scene), 4)

            meta_clean = copy.deepcopy(entry["metadata"])
            if "group_id" not in meta_clean:
                meta_clean["group_id"] = gid

            fused_results.append(
                {
                    "group_id": gid,
                    "score": fused_score,
                    "score_detail": {
                        "s_ship": round(s_ship, 4),
                        "s_scene": round(s_scene, 4),
                    },
                    "ship_crop_image_id": entry["ship_crop_image_id"],
                    "context_crop_image_id": entry["context_crop_image_id"],
                    "full_frame_image_id": entry["full_frame_image_id"],
                    "metadata": meta_clean,
                }
            )

        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return fused_results[:k]

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
