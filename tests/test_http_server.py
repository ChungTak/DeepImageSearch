# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Unit tests for DeepImageSearch FastAPI HTTP API Server."""

import io
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from DeepImageSearch.server.app import create_app
from DeepImageSearch.server.cli import build_parser
from DeepImageSearch.server.config import ServerConfig
from DeepImageSearch.server.state import EngineManager


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (48, 48), color=(200, 100, 50))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_engine(tmp_path: Path, sample_jpeg_bytes: bytes):
    img_file = tmp_path / "sunset_001.jpg"
    img_file.write_bytes(sample_jpeg_bytes)

    engine = MagicMock()
    engine.count = 1
    engine.info.return_value = {
        "model_name": "clip-vit-b-32",
        "vector_store": "faiss",
        "dimension": 512,
        "indexed_count": 1,
    }
    sample_record = {
        "image_id": "rec_001",
        "image_index": 0,
        "image_name": "sunset_001.jpg",
        "image_path": str(img_file),
        "caption": "A golden sunset",
        "indexed_at": "2026-10-23T00:00:00+00:00",
    }
    engine.get_records.return_value = [sample_record]
    engine.get_record.side_effect = lambda rid: sample_record if rid == "rec_001" else None

    def fake_index(paths, **kwargs):
        added = len(paths) if isinstance(paths, list) else 1
        engine.count += added
        return added

    def fake_delete(ids, **kwargs):
        id_list = [ids] if isinstance(ids, str) else list(ids)
        deleted = sum(1 for rid in id_list if rid == "rec_001")
        engine.count = max(0, engine.count - deleted)
        return deleted

    def fake_clear(**kwargs):
        cleared = engine.count
        engine.count = 0
        return cleared

    engine.index.side_effect = fake_index
    engine.delete.side_effect = fake_delete
    engine.clear.side_effect = fake_clear
    engine.search.return_value = [
        {
            "id": "rec_001",
            "score": 0.91,
            "metadata": sample_record,
        }
    ]
    return engine


@pytest.fixture
def api_client(tmp_path: Path, mock_engine: MagicMock) -> TestClient:
    cfg = ServerConfig(
        index_path=str(tmp_path / "index"),
        upload_dir=str(tmp_path / "uploads"),
        api_key="test-secret-key",
    )
    mgr = EngineManager(config=cfg, engine=mock_engine)
    app = create_app(config=cfg, engine_manager=mgr)
    return TestClient(app)


def test_health_endpoint_requires_no_auth(api_client: TestClient):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["indexed_count"] == 1


def test_auth_protection_on_v1_routes(api_client: TestClient):
    resp = api_client.get("/api/v1/info")
    assert resp.status_code == 401

    resp_ok = api_client.get("/api/v1/info", headers={"X-API-Key": "test-secret-key"})
    assert resp_ok.status_code == 200
    assert resp_ok.json()["dimension"] == 512


def test_index_paths_endpoint(api_client: TestClient, tmp_path: Path, sample_jpeg_bytes: bytes):
    img1 = tmp_path / "local1.jpg"
    img2 = tmp_path / "local2.jpg"
    img1.write_bytes(sample_jpeg_bytes)
    img2.write_bytes(sample_jpeg_bytes)

    resp = api_client.post(
        "/api/v1/index/paths",
        json={"paths": [str(img1), str(img2)], "generate_captions": False},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "indexed"
    assert body["added_count"] == 2


def test_upload_and_index_images(api_client: TestClient, sample_jpeg_bytes: bytes):
    files = [
        ("files", ("photo1.jpg", sample_jpeg_bytes, "image/jpeg")),
        ("files", ("photo2.jpg", sample_jpeg_bytes, "image/jpeg")),
    ]
    resp = api_client.post(
        "/api/v1/index/upload",
        files=files,
        data={"generate_captions": "false"},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "indexed"
    assert body["added_count"] == 2
    assert body["total_indexed"] >= 3


def test_index_save_and_load(api_client: TestClient, tmp_path: Path):
    headers = {"X-API-Key": "test-secret-key"}
    save_path = str(tmp_path / "saved_index")

    # 1. Save
    resp_save = api_client.post("/api/v1/index/save", json={"index_path": save_path}, headers=headers)
    assert resp_save.status_code == 200
    assert resp_save.json()["status"] == "saved"

    # 2. Load
    resp_load = api_client.post("/api/v1/index/load", json={"index_path": save_path}, headers=headers)
    assert resp_load.status_code == 200
    assert resp_load.json()["status"] == "loaded"

    # 3. Load non-existent
    resp_err = api_client.post("/api/v1/index/load", json={"index_path": "/no/such/path"}, headers=headers)
    assert resp_err.status_code == 404


def test_text_image_and_hybrid_search(api_client: TestClient, sample_jpeg_bytes: bytes):
    headers = {"X-API-Key": "test-secret-key"}

    # 1. Text search
    r_text = api_client.post(
        "/api/v1/search/text",
        json={"query": "golden sunset", "k": 5},
        headers=headers,
    )
    assert r_text.status_code == 200
    assert r_text.json()["query_mode"] == "text"
    assert r_text.json()["results"][0]["id"] == "rec_001"

    # 2. Image search via file upload
    r_img = api_client.post(
        "/api/v1/search/image",
        files={"file": ("query.jpg", sample_jpeg_bytes, "image/jpeg")},
        data={"k": "5"},
        headers=headers,
    )
    assert r_img.status_code == 200
    assert r_img.json()["query_mode"] == "image"

    # 3. Hybrid search via text + file upload
    r_hybrid = api_client.post(
        "/api/v1/search/hybrid",
        files={"file": ("query.jpg", sample_jpeg_bytes, "image/jpeg")},
        data={
            "text_query": "sunset over ocean",
            "text_weight": "0.6",
            "mode": "hybrid",
            "k": "5",
        },
        headers=headers,
    )
    assert r_hybrid.status_code == 200
    assert r_hybrid.json()["query_mode"] == "hybrid"


def test_search_validation_errors(api_client: TestClient):
    headers = {"X-API-Key": "test-secret-key"}

    # Image search without file or path
    resp_img = api_client.post("/api/v1/search/image", headers=headers)
    assert resp_img.status_code == 400

    # Image search with non-existent path
    resp_path = api_client.post("/api/v1/search/image", data={"image_path": "/missing.jpg"}, headers=headers)
    assert resp_path.status_code == 404

    # Hybrid search without image
    resp_hyb = api_client.post("/api/v1/search/hybrid", data={"text_query": "test"}, headers=headers)
    assert resp_hyb.status_code == 400


def test_records_and_image_content_streaming(api_client: TestClient):
    headers = {"X-API-Key": "test-secret-key"}

    r_list = api_client.get("/api/v1/records?offset=0&limit=10", headers=headers)
    assert r_list.status_code == 200
    assert r_list.json()["total"] == 1

    r_detail = api_client.get("/api/v1/records/rec_001", headers=headers)
    assert r_detail.status_code == 200
    assert r_detail.json()["image_name"] == "sunset_001.jpg"

    r_content = api_client.get("/api/v1/records/rec_001/content", headers=headers)
    assert r_content.status_code == 200
    assert r_content.headers["content-type"].startswith("image/")

    # Non-existent record
    r_404 = api_client.get("/api/v1/records/rec_missing", headers=headers)
    assert r_404.status_code == 404


def test_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--port", "9000", "--model", "clip-vit-l-14", "--api-key", "my-key"])
    assert args.port == 9000
    assert args.model == "clip-vit-l-14"
    assert args.api_key == "my-key"


def test_dual_search_endpoint_basic(api_client: TestClient, sample_jpeg_bytes: bytes):
    headers = {"X-API-Key": "test-secret-key"}

    # 1. Dual search with natural language query
    resp = api_client.post(
        "/api/v1/search/dual",
        data={
            "query": "夜间停靠在码头边的红色小型渔船",
            "w1": "0.7",
            "w2": "0.3",
            "k": "10",
        },
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_mode"] == "dual_fusion"
    assert body["count"] >= 1
    first = body["results"][0]
    assert "group_id" in first
    assert "score" in first
    assert "score_detail" in first
    assert "s_ship" in first["score_detail"]
    assert "s_scene" in first["score_detail"]


def test_dual_search_with_crop_alignment(tmp_path: Path):
    headers = {"X-API-Key": "test-secret-key"}

    # Setup a mock engine returning different results for ship vs scene
    engine = MagicMock()
    engine.count = 2
    engine.info.return_value = {"model_name": "mock", "vector_store": "faiss", "dimension": 128, "indexed_count": 2}

    def fake_search(query, **kwargs):
        filters = kwargs.get("filters") or {}
        # Simulate ship branch vs scene branch
        if "crop_type" in filters and filters["crop_type"] == "ship_crop":
            return [
                {
                    "id": "fe4a77edc6599079df4678e011ebaf93",
                    "score": 0.942,
                    "metadata": {
                        "group_id": "GRP_20261022_001",
                        "crop_type": "ship_crop",
                        "ship_crop_image_id": "fe4a77edc6599079df4678e011ebaf93",
                        "camera_id": "CAM_DOCK_03",
                        "camera_location": "东港区3号货运码头",
                        "ship_type": "fishing_boat",
                        "ship_size": "small",
                        "ship_color": "red",
                        "color_confidence": "high",
                        "state": "stationary",
                        "time_of_day": "night",
                        "scene_zone": "dock",
                        "start_time": "2026-10-22 21:15:00",
                        "duration_sec": 2100,
                    },
                }
            ]
        else:
            return [
                {
                    "id": "a1b2c3d4e5f6789012345678abcdef01",
                    "score": 0.862,
                    "metadata": {
                        "group_id": "GRP_20261022_001",
                        "crop_type": "full_frame",
                        "context_crop_image_id": "77bc88edc6599079df4678e011eb0012",
                        "full_frame_image_id": "a1b2c3d4e5f6789012345678abcdef01",
                        "camera_id": "CAM_DOCK_03",
                    },
                }
            ]

    engine.search.side_effect = fake_search
    # Add metadata to simulate crop_type detection
    engine.vector_store = MagicMock()
    engine.vector_store._metadata = [{"crop_type": "ship_crop"}, {"crop_type": "full_frame"}]

    cfg = ServerConfig(
        index_path=str(tmp_path / "index_dual"),
        upload_dir=str(tmp_path / "uploads_dual"),
        api_key="test-secret-key",
    )
    mgr = EngineManager(config=cfg, engine=engine)
    client = TestClient(create_app(config=cfg, engine_manager=mgr))

    resp = client.post(
        "/api/v1/search/dual",
        data={
            "query": "夜间停靠在码头边的红色小型渔船",
            "ship_query": "红色小型渔船",
            "scene_query": "夜间码头泊位",
            "w1": "0.7",
            "w2": "0.3",
            "filters_json": '{"duration_sec": {"$gte": 1200}, "ship_size": {"$in": ["small", "medium"]}}',
            "night_color_tolerance": "true",
            "k": "20",
        },
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_mode"] == "dual_fusion"
    assert data["count"] == 1
    res = data["results"][0]
    assert res["group_id"] == "GRP_20261022_001"
    # 0.7 * 0.942 + 0.3 * 0.862 = 0.6594 + 0.2586 = 0.918
    assert res["score"] == 0.918
    assert res["score_detail"]["s_ship"] == 0.942
    assert res["score_detail"]["s_scene"] == 0.862
    assert res["ship_crop_image_id"] == "fe4a77edc6599079df4678e011ebaf93"
    assert res["context_crop_image_id"] == "77bc88edc6599079df4678e011eb0012"
    assert res["full_frame_image_id"] == "a1b2c3d4e5f6789012345678abcdef01"
    assert res["metadata"]["ship_type"] == "fishing_boat"
    assert res["metadata"]["camera_location"] == "东港区3号货运码头"


def test_dual_search_validation_and_errors(api_client: TestClient):
    headers = {"X-API-Key": "test-secret-key"}

    # No query, ship_query, scene_query, or file
    r_empty = api_client.post("/api/v1/search/dual", headers=headers)
    assert r_empty.status_code == 400

    # Invalid filters_json
    r_invalid_json = api_client.post(
        "/api/v1/search/dual",
        data={"query": "test query", "filters_json": "{invalid-json"},
        headers=headers,
    )
    assert r_invalid_json.status_code == 400


def test_delete_single_record_endpoint(api_client: TestClient):
    headers = {"X-API-Key": "test-secret-key"}

    # 1. Successful deletion
    resp = api_client.delete("/api/v1/records/rec_001", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"
    assert data["image_id"] == "rec_001"
    assert data["deleted_count"] >= 1

    # 2. Deletion of non-existent record
    resp_404 = api_client.delete("/api/v1/records/rec_non_existent", headers=headers)
    assert resp_404.status_code == 404


def test_delete_batch_endpoint(api_client: TestClient):
    headers = {"X-API-Key": "test-secret-key"}

    # 1. Delete by image_ids
    resp = api_client.post(
        "/api/v1/index/delete",
        json={"image_ids": ["rec_001"]},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"
    assert "rec_001" in data["deleted_ids"]

    # 2. Delete without ids or filters raises 400
    resp_bad = api_client.post(
        "/api/v1/index/delete",
        json={},
        headers=headers,
    )
    assert resp_bad.status_code == 400


def test_delete_with_filters_endpoint(api_client: TestClient):
    headers = {"X-API-Key": "test-secret-key"}

    resp = api_client.post(
        "/api/v1/index/delete",
        json={"filters": {"caption": "A golden sunset"}},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"
    assert "rec_001" in data["deleted_ids"]


def test_clear_index_endpoints(api_client: TestClient):
    headers = {"X-API-Key": "test-secret-key"}

    # POST /api/v1/index/clear
    resp = api_client.post("/api/v1/index/clear", json={"save_after_clear": True}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cleared"
    assert data["total_indexed"] == 0

    # DELETE /api/v1/index
    resp_del = api_client.delete("/api/v1/index", headers=headers)
    assert resp_del.status_code == 200
    assert resp_del.json()["status"] == "cleared"


def test_real_engine_deletion_integration(tmp_path: Path, monkeypatch, sample_jpeg_bytes: bytes):
    from DeepImageSearch import SearchEngine
    from tests.conftest import DummyEmbedding

    monkeypatch.setattr(
        "DeepImageSearch.core.embeddings.EmbeddingManager.create",
        staticmethod(lambda *args, **kwargs: DummyEmbedding()),
    )
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img3 = tmp_path / "img3.jpg"
    img1.write_bytes(sample_jpeg_bytes)
    img2.write_bytes(sample_jpeg_bytes)
    img3.write_bytes(sample_jpeg_bytes)

    engine = SearchEngine(index_dir=str(tmp_path / "engine_idx"))
    cfg = ServerConfig(index_path=str(tmp_path / "engine_idx"), upload_dir=str(tmp_path / "uploads"), api_key="secret")
    mgr = EngineManager(config=cfg, engine=engine)
    client = TestClient(create_app(config=cfg, engine_manager=mgr))
    headers = {"X-API-Key": "secret"}

    # 1. Index 3 images with metadata
    r_idx = client.post(
        "/api/v1/index/paths",
        json={
            "paths": [str(img1), str(img2), str(img3)],
            "metadata": [
                {"state": "active", "group": "g1"},
                {"state": "inactive", "group": "g1"},
                {"state": "inactive", "group": "g2"},
            ],
        },
        headers=headers,
    )
    assert r_idx.status_code == 200
    assert mgr.count == 3

    # 2. Delete 1 by ID
    records = mgr.engine.get_records()
    first_id = records[0]["image_id"]
    r_del1 = client.delete(f"/api/v1/records/{first_id}", headers=headers)
    assert r_del1.status_code == 200
    assert mgr.count == 2
    assert client.get(f"/api/v1/records/{first_id}", headers=headers).status_code == 404

    # 3. Delete by metadata filter
    r_del_filter = client.post(
        "/api/v1/index/delete",
        json={"filters": {"state": "inactive"}},
        headers=headers,
    )
    assert r_del_filter.status_code == 200
    assert r_del_filter.json()["deleted_count"] >= 1

    # 4. Clear remaining
    r_clear = client.post("/api/v1/index/clear", headers=headers)
    assert r_clear.status_code == 200
    assert r_clear.json()["total_indexed"] == 0
    assert mgr.count == 0


