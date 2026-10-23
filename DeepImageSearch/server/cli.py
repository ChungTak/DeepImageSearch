# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Command line interface for DeepImageSearch HTTP API Server."""

import argparse
from typing import Optional

try:
    import uvicorn
except ImportError as exc:
    raise ImportError(
        "uvicorn is required to run the HTTP server. Install with: pip install 'DeepImageSearch[api]'"
    ) from exc

from DeepImageSearch.server.app import create_app
from DeepImageSearch.server.config import ServerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-image-search-server",
        description="Start the DeepImageSearch RESTful HTTP API Server.",
    )
    parser.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 8000)")
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model preset or name (default: clip-vit-b-32 or embeddinggemma-2)",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Directory to load/save vector index and metadata (default: ./dis_index)",
    )
    parser.add_argument(
        "--vector-store",
        default=None,
        choices=["faiss", "chroma", "qdrant"],
        help="Vector store backend (default: faiss)",
    )
    parser.add_argument(
        "--upload-dir",
        default=None,
        help="Directory to persist uploaded indexed images (default: ./dis_uploads)",
    )
    parser.add_argument("--device", default=None, help="Compute device: cuda, mps, or cpu")
    parser.add_argument("--api-key", default=None, help="Optional X-API-Key for authentication")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    return parser


def main(args_list: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(args_list)

    cfg = ServerConfig()
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port
    if args.model is not None:
        cfg.model_name = args.model
    if args.index_path is not None:
        cfg.index_path = args.index_path
    if args.vector_store is not None:
        cfg.vector_store = args.vector_store
    if args.upload_dir is not None:
        cfg.upload_dir = args.upload_dir
    if args.device is not None:
        cfg.device = args.device
    if args.api_key is not None:
        cfg.api_key = args.api_key

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, reload=args.reload)


if __name__ == "__main__":
    main()
