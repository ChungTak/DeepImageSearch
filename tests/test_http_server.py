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

    engine.index.side_effect = fake_index
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
