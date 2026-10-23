# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Unit tests for EmbeddingGemma 2 (google/embeddinggemma-2) multimodal embedding backend.

Follows the repository CI standard: completely mocked, no network downloads.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
from PIL import Image

from DeepImageSearch.core.embeddings import (
    EmbeddingGemma2Embedding,
    EmbeddingManager,
    GEMMA2_PRESETS,
)
from DeepImageSearch.core.searcher import Searcher
from DeepImageSearch.search_engine import SearchEngine
from DeepImageSearch.DeepImageSearch import Search_Setup


@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (64, 64), color=(120, 180, 240))
    img.save(img_path)
    return img_path


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_default_init_skips_audio_encoder(mock_st_cls: MagicMock):
    """验证默认初始化跳过 300M 音频编码器且维度为 768。"""
    emb = EmbeddingManager.create("embeddinggemma-2", device="cpu")
    assert isinstance(emb, EmbeddingGemma2Embedding)
    assert emb.dimension == 768
    assert emb.supports_text is True

    mock_st_cls.assert_called_once()
    _, call_kwargs = mock_st_cls.call_args
    assert call_kwargs["config_kwargs"] == {"audio_config": None}
    assert call_kwargs["truncate_dim"] is None
    assert call_kwargs["model_kwargs"]["torch_dtype"] == torch.float32


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_mrl_truncation_preset(mock_st_cls: MagicMock):
    """验证 MRL 256d 预设正确设置 truncate_dim=256。"""
    emb = EmbeddingManager.create("embeddinggemma-2-256d", device="cpu")
    assert emb.dimension == 256
    _, call_kwargs = mock_st_cls.call_args
    assert call_kwargs["truncate_dim"] == 256


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_rejects_float16(mock_st_cls: MagicMock):
    """验证传入 float16 时立即抛出 ValueError，防止 NaN 输出。"""
    with pytest.raises(ValueError, match="float16"):
        EmbeddingGemma2Embedding(device="cpu", torch_dtype=torch.float16)


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_rejects_invalid_mrl_dim(mock_st_cls: MagicMock):
    """验证非法 MRL 维度抛出 ValueError。"""
    with pytest.raises(ValueError, match="Invalid truncate_dim"):
        EmbeddingGemma2Embedding(device="cpu", truncate_dim=300)


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_rejects_invalid_soft_tokens(mock_st_cls: MagicMock):
    """验证非法 soft tokens 预算抛出 ValueError。"""
    with pytest.raises(ValueError, match="Invalid max_soft_tokens"):
        EmbeddingGemma2Embedding(device="cpu", max_soft_tokens=100)


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_text_and_image_encoding_contracts(
    mock_st_cls: MagicMock, dummy_image: Path
):
    """验证文本编码自动带 SearchQuery 前缀，而图像编码封装为 {'image': PIL.Image} 且不带前缀。"""
    mock_instance = MagicMock()
    mock_instance.encode.side_effect = lambda inputs, **kwargs: (
        np.ones((len(inputs), 256), dtype=np.float32)
        if isinstance(inputs, list)
        else np.ones((256,), dtype=np.float32)
    )
    mock_st_cls.return_value = mock_instance

    emb = EmbeddingGemma2Embedding(
        device="cpu",
        truncate_dim=256,
        max_soft_tokens=560,
    )

    # 1. 测试文本查询编码
    text_vec = emb.embed_text("a red car parked near a lake")
    assert text_vec.shape == (256,)
    assert text_vec.dtype == np.float32
    text_call_args, text_call_kwargs = mock_instance.encode.call_args
    assert text_call_args[0] == "a red car parked near a lake"
    assert text_call_kwargs["prompt_name"] == "SearchQuery"
    assert text_call_kwargs["normalize_embeddings"] is True
    assert text_call_kwargs["truncate_dim"] == 256

    # 2. 测试单张图像编码（不得包含 prompt_name，且必须包含 processing_kwargs）
    img_vec = emb.embed_image(dummy_image)
    assert img_vec.shape == (256,)
    img_call_args, img_call_kwargs = mock_instance.encode.call_args
    assert "image" in img_call_args[0]
    assert isinstance(img_call_args[0]["image"], Image.Image)
    assert "prompt_name" not in img_call_kwargs
    assert img_call_kwargs["normalize_embeddings"] is True
    assert img_call_kwargs["processing_kwargs"] == {"image": {"max_soft_tokens": 560}}

    # 3. 测试批量图像编码
    batch_vecs = emb.embed_images([dummy_image, dummy_image], batch_size=8)
    assert batch_vecs.shape == (2, 256)

    # 4. 测试批量文本编码
    batch_texts = emb.embed_texts(["query 1", "query 2"], batch_size=8)
    assert batch_texts.shape == (2, 256)

    # 5. 测试交错多模态编码
    mm_vec = emb.embed_multimodal("waterproof trail shoe", dummy_image)
    assert mm_vec.shape == (256,)
    mm_call_args, _ = mock_instance.encode.call_args
    assert "<|image|>" in mm_call_args[0]["text"]
    assert isinstance(mm_call_args[0]["image"], Image.Image)


@patch("sentence_transformers.SentenceTransformer")
def test_v2_search_setup_compat_strips_legacy_kwargs(
    mock_st_cls: MagicMock, dummy_image: Path
):
    """验证 v2 Search_Setup 传入 pretrained 和 image_size 时不会导致 EmbeddingGemma2 报错。"""
    emb = EmbeddingManager.create(
        "google/embeddinggemma-2",
        device="cpu",
        pretrained=True,
        image_size=224,
    )
    assert isinstance(emb, EmbeddingGemma2Embedding)
    assert emb.dimension == 768


@patch("sentence_transformers.SentenceTransformer")
def test_v2_search_setup_safe_index_dir(mock_st_cls: MagicMock, tmp_path: Path, dummy_image: Path):
    """验证 v2 Search_Setup 使用包含斜杠的模型名称时正确转义目录。"""
    setup = Search_Setup(
        image_list=[str(dummy_image)],
        model_name="google/embeddinggemma-2",
        metadata_dir=str(tmp_path),
    )
    assert setup._index_dir == str(tmp_path / "google_embeddinggemma-2")
    assert setup.embedding.dimension == 768


@patch("sentence_transformers.SentenceTransformer")
def test_searchengine_kwargs_passthrough(mock_st_cls: MagicMock):
    """验证 SearchEngine 可透传 truncate_dim 与 max_soft_tokens 给 EmbeddingGemma 2。"""
    engine = SearchEngine(
        model_name="embeddinggemma-2",
        device="cpu",
        truncate_dim=256,
        max_soft_tokens=280,
    )
    assert isinstance(engine.embedding, EmbeddingGemma2Embedding)
    assert engine.embedding.dimension == 256
    info = engine.info()
    assert info["model"] == "embeddinggemma-2"
    assert info["dimension"] == 256


@patch("sentence_transformers.SentenceTransformer")
def test_searcher_hybrid_and_interleaved_modes(mock_st_cls: MagicMock, dummy_image: Path):
    """验证 Searcher 支持默认加权向量融合与原生交错多模态查询模式。"""
    mock_instance = MagicMock()
    mock_instance.encode.side_effect = lambda inputs, **kwargs: (
        np.full((256,), 0.5, dtype=np.float32)
    )
    mock_st_cls.return_value = mock_instance

    emb = EmbeddingGemma2Embedding(device="cpu", truncate_dim=256)
    mock_store = MagicMock()
    mock_store.search.return_value = [{"id": "1", "score": 0.95, "metadata": {}}]

    searcher = Searcher(embedding=emb, vector_store=mock_store)

    # 1. 默认加权融合
    results = searcher.search(
        query="running shoe",
        image_query=str(dummy_image),
        mode="hybrid",
        text_weight=0.7,
    )
    assert len(results) == 1
    mock_store.search.assert_called_once()
    fused_vec = mock_store.search.call_args[0][0]
    assert fused_vec.shape == (256,)
    # Verify vector is L2 normalized
    np.testing.assert_allclose(np.linalg.norm(fused_vec), 1.0, atol=1e-5)

    # 2. 原生交错模式
    mock_store.reset_mock()
    results_interleaved = searcher.search(
        query="waterproof boots",
        image_query=str(dummy_image),
        mode="interleaved",
    )
    assert len(results_interleaved) == 1
    mock_store.search.assert_called_once()
    interleaved_vec = mock_store.search.call_args[0][0]
    assert interleaved_vec.shape == (256,)
    np.testing.assert_allclose(np.linalg.norm(interleaved_vec), 1.0, atol=1e-5)


def test_presets_table_completeness():
    """验证 GEMMA2_PRESETS 包含计划中的全部预设。"""
    assert "embeddinggemma-2" in GEMMA2_PRESETS
    assert "embeddinggemma-2-768d" in GEMMA2_PRESETS
    assert "embeddinggemma-2-512d" in GEMMA2_PRESETS
    assert "embeddinggemma-2-256d" in GEMMA2_PRESETS
    assert "embeddinggemma-2-128d" in GEMMA2_PRESETS
    assert GEMMA2_PRESETS["embeddinggemma-2-256d"]["truncate_dim"] == 256


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_load_audio_encoder_enabled(mock_st_cls: MagicMock):
    """验证 load_audio_encoder=True 时不设置 audio_config 为 None。"""
    emb = EmbeddingGemma2Embedding(device="cpu", load_audio_encoder=True)
    _, call_kwargs = mock_st_cls.call_args
    assert "audio_config" not in call_kwargs["config_kwargs"]
    assert emb.dimension == 768


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_empty_inputs(mock_st_cls: MagicMock):
    """验证空输入直接返回空数组而不调用模型。"""
    mock_instance = MagicMock()
    mock_st_cls.return_value = mock_instance
    emb = EmbeddingGemma2Embedding(device="cpu", truncate_dim=256)

    assert emb.embed_images([]).shape == (0, 256)
    assert emb.embed_texts([]).shape == (0, 256)
    mock_instance.encode.assert_not_called()


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_image_input_types_and_conversion(mock_st_cls: MagicMock, tmp_path: Path):
    """验证 RGBA 图像自动转为 RGB，且支持网络 URL。"""
    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.zeros((768,), dtype=np.float32)
    mock_st_cls.return_value = mock_instance
    emb = EmbeddingGemma2Embedding(device="cpu")

    # 1. RGBA Image
    rgba_img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    emb.embed_image(rgba_img)
    call_args, _ = mock_instance.encode.call_args
    assert call_args[0]["image"].mode == "RGB"

    # 2. URL image
    emb.embed_image("https://example.com/test.jpg")
    call_args, _ = mock_instance.encode.call_args
    assert call_args[0]["image"] == "https://example.com/test.jpg"

    # 3. Invalid type
    with pytest.raises(TypeError, match="Unsupported image input type"):
        emb.embed_image(12345)


@patch("sentence_transformers.SentenceTransformer")
def test_embeddinggemma2_custom_prompt_name(mock_st_cls: MagicMock):
    """验证自定义 prompt_name 生效。"""
    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.zeros((768,), dtype=np.float32)
    mock_st_cls.return_value = mock_instance
    emb = EmbeddingGemma2Embedding(device="cpu")

    emb.embed_text("document content", prompt_name="Document")
    _, call_kwargs = mock_instance.encode.call_args
    assert call_kwargs["prompt_name"] == "Document"

