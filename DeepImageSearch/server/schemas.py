# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Pydantic data schemas for DeepImageSearch HTTP API Server."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_name: str
    vector_store: str
    indexed_count: int


class IndexPathsRequest(BaseModel):
    paths: Union[str, List[str]] = Field(
        ...,
        description="Folder path or list of image file paths on the server to index.",
    )
    generate_captions: bool = Field(
        default=False,
        description="Whether to auto-generate captions using configured LLM captioner.",
    )
    metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None,
        description="Optional custom metadata dictionary or list of dictionaries per image.",
    )
    save_after_index: bool = Field(
        default=True,
        description="Automatically persist index to disk after indexing completes.",
    )


class IndexResponse(BaseModel):
    status: str = "indexed"
    added_count: int
    total_indexed: int
    index_path: str


class IndexPersistenceRequest(BaseModel):
    index_path: Optional[str] = Field(
        default=None,
        description="Optional override path for saving/loading the index.",
    )


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query.")
    k: int = Field(default=10, ge=1, le=200, description="Number of results to return.")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata key-value filter conditions."
    )


class SearchResultItem(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query_mode: str
    count: int
    results: List[SearchResultItem]


class RecordListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    records: List[Dict[str, Any]]
