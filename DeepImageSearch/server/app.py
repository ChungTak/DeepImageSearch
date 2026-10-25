# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""FastAPI HTTP Server application for DeepImageSearch."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

try:
    from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError as exc:
    raise ImportError(
        "HTTP Server requires 'fastapi', 'uvicorn', 'python-multipart', and 'pydantic'. "
        "Install them with: pip install 'DeepImageSearch[api]'"
    ) from exc

from DeepImageSearch.server.config import ServerConfig
from DeepImageSearch.server.schemas import (
    ClearIndexRequest,
    ClearIndexResponse,
    DeleteIndexRequest,
    DeleteIndexResponse,
    DeleteRecordResponse,
    DualSearchResponse,
    HealthResponse,
    IndexPathsRequest,
    IndexPersistenceRequest,
    IndexResponse,
    RecordListResponse,
    SearchResponse,
    TextSearchRequest,
)
from DeepImageSearch.server.state import ALLOWED_IMAGE_EXTENSIONS, EngineManager


def create_app(
    config: Optional[ServerConfig] = None,
    engine_manager: Optional[EngineManager] = None,
) -> FastAPI:
    """Create and configure the DeepImageSearch FastAPI application."""
    cfg = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "engine_manager", None) is None:
            app.state.engine_manager = EngineManager(cfg)
        yield

    app = FastAPI(
        title="DeepImageSearch HTTP API",
        description="RESTful API for multimodal text-to-image, image-to-image, and hybrid search.",
        version="3.0.3",
        lifespan=lifespan,
    )

    if engine_manager is not None:
        app.state.engine_manager = engine_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if cfg.api_key and x_api_key != cfg.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header.",
            )

    def get_manager() -> EngineManager:
        mgr = getattr(app.state, "engine_manager", None)
        if mgr is None:
            mgr = EngineManager(cfg)
            app.state.engine_manager = mgr
        return mgr

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health_check(mgr: EngineManager = Depends(get_manager)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_name=cfg.model_name,
            vector_store=cfg.vector_store,
            indexed_count=mgr.count,
        )

    @app.get(
        "/api/v1/info",
        response_model=Dict[str, Any],
        dependencies=[Depends(verify_api_key)],
        tags=["System"],
    )
    def engine_info(mgr: EngineManager = Depends(get_manager)) -> Dict[str, Any]:
        return mgr.info()

    @app.post(
        "/api/v1/index/paths",
        response_model=IndexResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Indexing"],
    )
    def index_server_paths(
        req: IndexPathsRequest,
        mgr: EngineManager = Depends(get_manager),
    ) -> IndexResponse:
        try:
            added = mgr.index_paths(
                paths=req.paths,
                generate_captions=req.generate_captions,
                metadata=req.metadata,
                save_after_index=req.save_after_index,
            )
            return IndexResponse(
                status="indexed",
                added_count=added,
                total_indexed=mgr.count,
                index_path=cfg.index_path,
            )
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post(
        "/api/v1/index/upload",
        response_model=IndexResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Indexing"],
    )
    def index_uploaded_images(
        files: List[UploadFile] = File(..., description="One or more image files to index."),
        generate_captions: bool = Form(default=False),
        metadata_json: Optional[str] = Form(
            default=None,
            description="Optional JSON object or array of metadata objects.",
        ),
        save_after_index: bool = Form(default=True),
        mgr: EngineManager = Depends(get_manager),
    ) -> IndexResponse:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        parsed_metadata = None
        if metadata_json:
            try:
                parsed_metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid metadata_json: {exc}"
                ) from exc

        saved_paths: List[str] = []
        try:
            for upload in files:
                raw_bytes = upload.file.read()
                if not raw_bytes:
                    continue
                saved_path = mgr.save_uploaded_file(upload.filename or "upload.jpg", raw_bytes)
                saved_paths.append(saved_path)

            if not saved_paths:
                raise ValueError("Uploaded files were empty.")

            added = mgr.index_paths(
                paths=saved_paths,
                generate_captions=generate_captions,
                metadata=parsed_metadata,
                save_after_index=save_after_index,
            )
            return IndexResponse(
                status="indexed",
                added_count=added,
                total_indexed=mgr.count,
                index_path=cfg.index_path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(
        "/api/v1/index/save",
        dependencies=[Depends(verify_api_key)],
        tags=["Indexing"],
    )
    def save_index(
        req: Optional[IndexPersistenceRequest] = None,
        mgr: EngineManager = Depends(get_manager),
    ) -> Dict[str, Any]:
        target = req.index_path if req else None
        saved_to = mgr.save(target)
        return {"status": "saved", "index_path": saved_to, "total_indexed": mgr.count}

    @app.post(
        "/api/v1/index/load",
        dependencies=[Depends(verify_api_key)],
        tags=["Indexing"],
    )
    def load_index(
        req: Optional[IndexPersistenceRequest] = None,
        mgr: EngineManager = Depends(get_manager),
    ) -> Dict[str, Any]:
        source = req.index_path if req else None
        try:
            loaded_from = mgr.load(source)
            return {"status": "loaded", "index_path": loaded_from, "total_indexed": mgr.count}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/api/v1/index/delete",
        response_model=DeleteIndexResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Indexing"],
    )
    def delete_from_index(
        req: DeleteIndexRequest,
        mgr: EngineManager = Depends(get_manager),
    ) -> DeleteIndexResponse:
        try:
            res = mgr.delete_records(
                image_ids=req.image_ids,
                filters=req.filters,
                save_after_delete=req.save_after_delete,
            )
            return DeleteIndexResponse(**res)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/api/v1/index/clear",
        response_model=ClearIndexResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Indexing"],
    )
    @app.delete(
        "/api/v1/index",
        response_model=ClearIndexResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Indexing"],
    )
    def clear_index(
        req: Optional[ClearIndexRequest] = None,
        mgr: EngineManager = Depends(get_manager),
    ) -> ClearIndexResponse:
        save_after = req.save_after_clear if req else True
        res = mgr.clear(save_after_clear=save_after)
        return ClearIndexResponse(**res)

    @app.post(
        "/api/v1/search/text",
        response_model=SearchResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Search"],
    )
    def search_by_text(
        req: TextSearchRequest,
        mgr: EngineManager = Depends(get_manager),
    ) -> SearchResponse:
        try:
            results = mgr.search(
                query=req.query,
                mode="text",
                k=req.k,
                filters=req.filters,
            )
            return SearchResponse(query_mode="text", count=len(results), results=results)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/api/v1/search/image",
        response_model=SearchResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Search"],
    )
    def search_by_image(
        file: Optional[UploadFile] = File(default=None, description="Uploaded query image file."),
        image_path: Optional[str] = Form(default=None, description="Server-side image file path."),
        k: int = Form(default=10, ge=1, le=200),
        filters_json: Optional[str] = Form(default=None),
        mgr: EngineManager = Depends(get_manager),
    ) -> SearchResponse:
        if file is None and not image_path:
            raise HTTPException(
                status_code=400,
                detail="Provide either an uploaded 'file' or a server-side 'image_path'.",
            )

        filters = None
        if filters_json:
            try:
                filters = json.loads(filters_json)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid filters_json: {exc}") from exc

        temp_path: Optional[str] = None
        try:
            if file is not None:
                ext = Path(file.filename or "query.jpg").suffix.lower() or ".jpg"
                if ext not in ALLOWED_IMAGE_EXTENSIONS:
                    raise HTTPException(status_code=400, detail=f"Unsupported image extension: {ext}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file.file.read())
                    temp_path = tmp.name
                target_image = temp_path
            else:
                if not os.path.exists(image_path):
                    raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
                target_image = image_path

            results = mgr.search(
                image_query=target_image,
                mode="image",
                k=k,
                filters=filters,
            )
            return SearchResponse(query_mode="image", count=len(results), results=results)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @app.post(
        "/api/v1/search/hybrid",
        response_model=SearchResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Search"],
    )
    def search_hybrid(
        text_query: str = Form(..., min_length=1, description="Text query for hybrid search."),
        file: Optional[UploadFile] = File(default=None, description="Uploaded query image file."),
        image_path: Optional[str] = Form(default=None, description="Server-side image file path."),
        text_weight: float = Form(default=0.5, ge=0.0, le=1.0),
        mode: str = Form(default="hybrid", description="'hybrid' or 'interleaved'"),
        k: int = Form(default=10, ge=1, le=200),
        filters_json: Optional[str] = Form(default=None),
        mgr: EngineManager = Depends(get_manager),
    ) -> SearchResponse:
        if file is None and not image_path:
            raise HTTPException(
                status_code=400,
                detail="Hybrid search requires both 'text_query' and either 'file' or 'image_path'.",
            )

        filters = None
        if filters_json:
            try:
                filters = json.loads(filters_json)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid filters_json: {exc}") from exc

        temp_path: Optional[str] = None
        try:
            if file is not None:
                ext = Path(file.filename or "query.jpg").suffix.lower() or ".jpg"
                if ext not in ALLOWED_IMAGE_EXTENSIONS:
                    raise HTTPException(status_code=400, detail=f"Unsupported image extension: {ext}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file.file.read())
                    temp_path = tmp.name
                target_image = temp_path
            else:
                if not os.path.exists(image_path):
                    raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
                target_image = image_path

            results = mgr.search(
                query=text_query,
                image_query=target_image,
                mode=mode,
                k=k,
                text_weight=text_weight,
                filters=filters,
            )
            return SearchResponse(query_mode=mode, count=len(results), results=results)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @app.post(
        "/api/v1/search/dual",
        response_model=DualSearchResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Search"],
    )
    def search_dual(
        query: Optional[str] = Form(default=None, description="Natural language search query."),
        ship_query: Optional[str] = Form(default=None, description="Optional ship crop query."),
        scene_query: Optional[str] = Form(default=None, description="Optional scene query."),
        file: Optional[UploadFile] = File(default=None, description="Uploaded query image file."),
        filters_json: Optional[str] = Form(default=None, description="Structured filter JSON string."),
        w1: float = Form(default=0.7, description="Ship crop retrieval weight."),
        w2: float = Form(default=0.3, description="Scene retrieval weight."),
        night_color_tolerance: bool = Form(default=True, description="Enable night color tolerance."),
        k: int = Form(default=20, ge=1, le=200, description="Top-k results count."),
        mgr: EngineManager = Depends(get_manager),
    ) -> DualSearchResponse:
        if not query and not ship_query and not scene_query and file is None:
            raise HTTPException(
                status_code=400,
                detail="Dual search requires at least one of 'query', 'ship_query', 'scene_query', or uploaded 'file'.",
            )

        filters = None
        if filters_json:
            try:
                filters = json.loads(filters_json)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid filters_json: {exc}"
                ) from exc

        temp_path: Optional[str] = None
        try:
            if file is not None:
                ext = Path(file.filename or "query.jpg").suffix.lower() or ".jpg"
                if ext not in ALLOWED_IMAGE_EXTENSIONS:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported image extension: {ext}"
                    )
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file.file.read())
                    temp_path = tmp.name

            results = mgr.search_dual(
                query=query,
                ship_query=ship_query,
                scene_query=scene_query,
                image_query=temp_path,
                filters=filters,
                w1=w1,
                w2=w2,
                night_color_tolerance=night_color_tolerance,
                k=k,
            )
            return DualSearchResponse(query_mode="dual_fusion", count=len(results), results=results)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @app.get(
        "/api/v1/records",
        response_model=RecordListResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Records"],
    )
    def list_records(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=500),
        mgr: EngineManager = Depends(get_manager),
    ) -> RecordListResponse:
        data = mgr.get_records(offset=offset, limit=limit)
        return RecordListResponse(**data)

    @app.get(
        "/api/v1/records/{image_id}",
        response_model=Dict[str, Any],
        dependencies=[Depends(verify_api_key)],
        tags=["Records"],
    )
    def get_record_detail(
        image_id: str,
        mgr: EngineManager = Depends(get_manager),
    ) -> Dict[str, Any]:
        rec = mgr.get_record(image_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Record not found: {image_id}")
        return rec

    @app.get(
        "/api/v1/records/{image_id}/content",
        dependencies=[Depends(verify_api_key)],
        tags=["Records"],
    )
    def get_record_image_content(
        image_id: str,
        mgr: EngineManager = Depends(get_manager),
    ) -> FileResponse:
        rec = mgr.get_record(image_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Record not found: {image_id}")
        img_path = rec.get("image_path")
        if not img_path or not os.path.isfile(img_path):
            raise HTTPException(
                status_code=404,
                detail=f"Underlying image file not found on server for record {image_id}.",
            )
        return FileResponse(img_path)

    @app.delete(
        "/api/v1/records/{image_id}",
        response_model=DeleteRecordResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["Records", "Indexing"],
    )
    def delete_record(
        image_id: str,
        save_after_delete: bool = Query(default=True, description="Persist index after deletion."),
        mgr: EngineManager = Depends(get_manager),
    ) -> DeleteRecordResponse:
        try:
            res = mgr.delete_record(image_id=image_id, save_after_delete=save_after_delete)
            return DeleteRecordResponse(**res)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app
