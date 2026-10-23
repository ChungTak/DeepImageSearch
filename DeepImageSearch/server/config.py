# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Configuration for DeepImageSearch HTTP API Server."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ServerConfig:
    """Configuration for DeepImageSearch HTTP API Server."""

    host: str = field(default_factory=lambda: os.getenv("DIS_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("DIS_PORT", "8000")))
    model_name: str = field(
        default_factory=lambda: os.getenv("DIS_MODEL_NAME", "clip-vit-b-32")
    )
    vector_store: str = field(
        default_factory=lambda: os.getenv("DIS_VECTOR_STORE", "faiss")
    )
    index_path: str = field(
        default_factory=lambda: os.getenv("DIS_INDEX_PATH", "./dis_index")
    )
    upload_dir: str = field(
        default_factory=lambda: os.getenv("DIS_UPLOAD_DIR", "./dis_uploads")
    )
    device: Optional[str] = field(
        default_factory=lambda: os.getenv("DIS_DEVICE", None)
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("DIS_API_KEY", None)
    )
    cors_origins: List[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("DIS_CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]
    )
    captioner_model: Optional[str] = field(
        default_factory=lambda: os.getenv("DIS_CAPTIONER_MODEL", None)
    )
    captioner_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("DIS_CAPTIONER_API_KEY", None)
    )
    captioner_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("DIS_CAPTIONER_BASE_URL", None)
    )
