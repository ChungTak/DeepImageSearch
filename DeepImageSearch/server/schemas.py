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


class DeleteIndexRequest(BaseModel):
    image_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of image IDs to delete from the index.",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters to match images for deletion.",
    )
    save_after_delete: bool = Field(
        default=True,
        description="Automatically persist index changes to disk after deletion.",
    )


class DeleteIndexResponse(BaseModel):
    status: str = "deleted"
    deleted_count: int = Field(..., description="Number of images removed from index.")
    total_indexed: int = Field(..., description="Number of remaining indexed images.")
    deleted_ids: List[str] = Field(default_factory=list, description="IDs of deleted images.")


class DeleteRecordResponse(BaseModel):
    status: str = "deleted"
    image_id: str = Field(..., description="ID of the deleted image.")
    deleted_count: int = Field(default=1, description="Number of deleted records.")
    total_indexed: int = Field(..., description="Remaining total indexed images.")


class ClearIndexRequest(BaseModel):
    save_after_clear: bool = Field(
        default=True,
        description="Automatically persist emptied index state to disk.",
    )


class ClearIndexResponse(BaseModel):
    status: str = "cleared"
    deleted_count: int = Field(..., description="Number of purged images.")
    total_indexed: int = Field(default=0, description="Total images remaining after clearing.")



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


class ScoreDetail(BaseModel):
    s_ship: float = Field(..., description="Ship crop retrieval similarity score.")
    s_scene: float = Field(
        ..., description="Full frame / context scene retrieval similarity score."
    )


class DualSearchResultItem(BaseModel):
    group_id: str = Field(
        ...,
        description="Identifier for the aligned vessel track or detection group.",
    )
    score: float = Field(
        ...,
        description="Fused similarity score calculated as w1 * s_ship + w2 * s_scene.",
    )
    score_detail: ScoreDetail = Field(
        ..., description="Individual branch retrieval scores."
    )
    ship_crop_image_id: Optional[str] = Field(
        default=None, description="Image ID of the matched vessel crop."
    )
    context_crop_image_id: Optional[str] = Field(
        default=None, description="Image ID of the context crop."
    )
    full_frame_image_id: Optional[str] = Field(
        default=None, description="Image ID of the full frame image."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured attributes and detection metadata.",
    )


class DualSearchResponse(BaseModel):
    query_mode: str = Field(
        default="dual_fusion", description="Search execution mode."
    )
    count: int = Field(..., description="Number of grouped results returned.")
    results: List[DualSearchResultItem] = Field(
        ..., description="Ranked list of fused group search results."
    )

