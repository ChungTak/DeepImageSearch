# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Embedding backends for DeepImageSearch.

Supports:
- CLIP / SigLIP / EVA-CLIP via open_clip  (text + image)
- EmbeddingGemma 2 via sentence-transformers (text + image + interleaved multimodal)
- timm CNN/ViT models                     (image only, legacy)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

VALID_MRL_DIMS = (128, 256, 512, 768)
VALID_SOFT_TOKENS = (70, 140, 280, 560, 1120)

GEMMA2_PRESETS = {
    "embeddinggemma-2": {"model_id": "google/embeddinggemma-2", "truncate_dim": None},
    "embeddinggemma-2-768d": {"model_id": "google/embeddinggemma-2", "truncate_dim": 768},
    "embeddinggemma-2-512d": {"model_id": "google/embeddinggemma-2", "truncate_dim": 512},
    "embeddinggemma-2-256d": {"model_id": "google/embeddinggemma-2", "truncate_dim": 256},
    "embeddinggemma-2-128d": {"model_id": "google/embeddinggemma-2", "truncate_dim": 128},
}


class BaseEmbedding(ABC):
    """Abstract embedding backend."""

    supports_text: bool = False
    dimension: int = 0

    @abstractmethod
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """Return (N, D) float32 array of normalised image embeddings."""

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) float32 array of normalised text embeddings."""
        raise NotImplementedError("This embedding model does not support text queries")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Convenience: embed a single image and return 1-D vector."""
        return self.embed_images([image])[0]

    def embed_text(self, text: str) -> np.ndarray:
        """Convenience: embed a single text and return 1-D vector."""
        return self.embed_texts([text])[0]


class CLIPEmbedding(BaseEmbedding):
    """
    CLIP / SigLIP / EVA-CLIP embedding via open_clip.

    Supports both image and text embedding in a shared space.

    Parameters
    ----------
    model_name : str
        open_clip model name (e.g. 'ViT-B-32', 'ViT-L-14', 'ViT-bigG-14').
    pretrained : str
        Pretrained weights tag (e.g. 'laion2b_s34b_b79k', 'openai').
    device : str or None
        'cuda', 'mps', 'cpu', or None for auto-detect.
    batch_size : int
        Batch size for embedding extraction.
    """

    supports_text = True

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        import open_clip

        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        logger.info(f"Loading CLIP model: {model_name} (pretrained={pretrained}) on {device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Determine embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            self.dimension = self.model.encode_image(dummy).shape[-1]

        logger.info(f"CLIP model loaded — dimension={self.dimension}")

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            tensors = torch.stack([self.preprocess(img.convert("RGB")) for img in batch]).to(self.device)
            features = self.model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.vstack(all_features).astype(np.float32)

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        all_features = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.vstack(all_features).astype(np.float32)


class TimmEmbedding(BaseEmbedding):
    """
    Legacy timm-based image embedding (image-only).

    Parameters
    ----------
    model_name : str
        timm model name (e.g. 'vgg19', 'resnet50', 'vit_base_patch16_224').
    pretrained : bool
        Use pretrained weights.
    image_size : int
        Input image resolution.
    device : str or None
        Device for inference.
    batch_size : int
        Batch size for embedding extraction.
    """

    supports_text = False

    def __init__(
        self,
        model_name: str = "vgg19",
        pretrained: bool = True,
        image_size: int = 224,
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        import timm
        from torchvision import transforms

        self.image_size = image_size
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        logger.info(f"Loading timm model: {model_name} on {device}")
        base_model = timm.create_model(model_name, pretrained=pretrained)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()
        self.model.to(device)

        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Determine embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=device)
            self.dimension = self.model(dummy).flatten(1).shape[-1]

        logger.info(f"timm model loaded — dimension={self.dimension}")

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            tensors = torch.stack([self.preprocess(img.convert("RGB")) for img in batch]).to(self.device)
            features = self.model(tensors).flatten(1)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.vstack(all_features).astype(np.float32)


class CustomEmbedding(BaseEmbedding):
    """
    Custom embedding using any user-provided feature extractor.

    Accepts a callable that takes a PIL Image and returns a 1-D numpy array
    (or list/tensor). Works with ONNX, TorchScript, TensorFlow, sklearn,
    or any custom model.

    Parameters
    ----------
    extract_fn : callable
        Function that takes a PIL.Image and returns a feature vector
        (numpy array, list, or tensor). Must return the same dimension
        for every image.
    dimension : int
        Output dimension of the feature vector.
    name : str
        Name for this embedding (used in logging/metadata).

    Examples
    --------
    # ONNX model
    import onnxruntime as ort
    session = ort.InferenceSession("model.onnx")

    def my_extractor(img):
        arr = np.array(img.resize((224, 224))).astype(np.float32)
        arr = arr.transpose(2, 0, 1)[None] / 255.0
        return session.run(None, {"input": arr})[0].flatten()

    emb = CustomEmbedding(extract_fn=my_extractor, dimension=512)
    engine = SearchEngine(model_name="clip-vit-b-32")  # ignored
    engine.embedding = emb  # override

    # Or use directly with core modules
    from DeepImageSearch.core.indexer import Indexer
    from DeepImageSearch.vectorstores.faiss_store import FAISSStore

    store = FAISSStore(dimension=512)
    indexer = Indexer(embedding=emb, vector_store=store)
    indexer.index(image_paths)
    """

    supports_text = False

    def __init__(self, extract_fn: callable, dimension: int, name: str = "custom"):
        self.extract_fn = extract_fn
        self.dimension = dimension
        self.name = name
        logger.info(f"CustomEmbedding loaded — name={name}, dimension={dimension}")

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        features = []
        for img in images:
            feat = self.extract_fn(img.convert("RGB"))
            feat = np.asarray(feat, dtype=np.float32).flatten()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            features.append(feat)
        return np.vstack(features).astype(np.float32)


class EmbeddingGemma2Embedding(BaseEmbedding):
    """
    Google EmbeddingGemma 2 (google/embeddinggemma-2) multimodal embedding backend.

    Supports unified 768d (or MRL-truncated 512d/256d/128d) text, image, and
    interleaved multimodal embeddings via sentence-transformers (>=6.1.0).
    """

    supports_text: bool = True

    def __init__(
        self,
        model_id: str = "google/embeddinggemma-2",
        device: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        load_audio_encoder: bool = False,
        max_soft_tokens: Optional[int] = None,
        query_prompt_name: str = "SearchQuery",
        document_prompt_name: str = "Document",
        torch_dtype: Optional[Any] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "google/embeddinggemma-2 requires 'sentence-transformers>=6.1.0' and 'transformers'. "
                "Install them with: pip install 'DeepImageSearch[gemma]' or "
                "pip install -U 'sentence-transformers[image]>=6.1.0' transformers"
            ) from exc

        if truncate_dim is not None and truncate_dim not in VALID_MRL_DIMS:
            raise ValueError(
                f"Invalid truncate_dim={truncate_dim} for EmbeddingGemma 2. "
                f"Supported MRL dimensions are {VALID_MRL_DIMS}."
            )

        if max_soft_tokens is not None and max_soft_tokens not in VALID_SOFT_TOKENS:
            raise ValueError(
                f"Invalid max_soft_tokens={max_soft_tokens}. "
                f"Supported vision token budgets are {VALID_SOFT_TOKENS}."
            )

        self.model_id = model_id
        self.truncate_dim = truncate_dim
        self.dimension = truncate_dim if truncate_dim is not None else 768
        self.max_soft_tokens = max_soft_tokens
        self.query_prompt_name = query_prompt_name
        self.document_prompt_name = document_prompt_name
        self.batch_size = batch_size

        # 1. Resolve target device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # 2. Enforce numerical precision safety (NEVER allow float16)
        if torch_dtype is not None:
            if torch_dtype == torch.float16 or str(torch_dtype) in ("float16", "fp16", "torch.float16"):
                raise ValueError(
                    "EmbeddingGemma 2 does not support float16 due to activation range overflow (NaN). "
                    "Use torch.bfloat16 or torch.float32 instead."
                )
            resolved_dtype = torch_dtype
        else:
            if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
                resolved_dtype = torch.bfloat16
            else:
                resolved_dtype = torch.float32
        self.torch_dtype = resolved_dtype

        # 3. Selective encoder loading: skip 300M audio encoder by default for image search
        config_kwargs: Dict[str, Any] = kwargs.pop("config_kwargs", {})
        if not load_audio_encoder and "audio_config" not in config_kwargs:
            config_kwargs["audio_config"] = None

        model_kwargs: Dict[str, Any] = kwargs.pop("model_kwargs", {})
        model_kwargs.setdefault("torch_dtype", self.torch_dtype)

        logger.info(
            "Loading EmbeddingGemma2 (%s) on device=%s, dtype=%s, dim=%d, audio_encoder=%s",
            self.model_id,
            self.device,
            self.torch_dtype,
            self.dimension,
            load_audio_encoder,
        )

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            truncate_dim=self.truncate_dim,
            config_kwargs=config_kwargs,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    def _get_processing_kwargs(self) -> Optional[Dict[str, Any]]:
        if self.max_soft_tokens is not None:
            return {"image": {"max_soft_tokens": self.max_soft_tokens}}
        return None

    @staticmethod
    def _prepare_image_input(image: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Wrap image into SentenceTransformer's expected multimodal dict format {'image': <PIL.Image>}.
        Converting local paths to RGB PIL.Image ensures consistent channel handling.
        """
        if isinstance(image, (str, Path)):
            path_str = str(image)
            if path_str.startswith(("http://", "https://")):
                return {"image": path_str}
            with Image.open(path_str) as img:
                return {"image": img.convert("RGB")}
        elif isinstance(image, Image.Image):
            return {"image": image.convert("RGB")}
        raise TypeError(f"Unsupported image input type: {type(image)}")

    def embed_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Encode a single image into a 1D L2-normalized float32 numpy vector of shape (D,)."""
        item = self._prepare_image_input(image)
        encode_kwargs: Dict[str, Any] = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.truncate_dim
        proc_kwargs = self._get_processing_kwargs()
        if proc_kwargs is not None:
            encode_kwargs["processing_kwargs"] = proc_kwargs

        vec = self.model.encode(item, **encode_kwargs)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def embed_images(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a batch of images into a 2D L2-normalized float32 numpy array of shape (N, D)."""
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        items = [self._prepare_image_input(img) for img in images]
        bs = batch_size if batch_size is not None else self.batch_size
        encode_kwargs: Dict[str, Any] = {
            "batch_size": bs,
            "show_progress_bar": show_progress_bar,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.truncate_dim
        proc_kwargs = self._get_processing_kwargs()
        if proc_kwargs is not None:
            encode_kwargs["processing_kwargs"] = proc_kwargs

        vecs = self.model.encode(items, **encode_kwargs)
        return np.asarray(vecs, dtype=np.float32).reshape(len(images), self.dimension)

    def embed_text(
        self,
        text: Union[str, List[str]],
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode text query (or list of queries) using asymmetric task instruction prefix.
        Defaults to prompt_name='SearchQuery' ('task: search result | query: ').
        """
        is_single = isinstance(text, str)
        active_prompt = prompt_name if prompt_name is not None else self.query_prompt_name
        encode_kwargs: Dict[str, Any] = {
            "prompt_name": active_prompt,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.truncate_dim

        vecs = self.model.encode(text, **encode_kwargs)
        arr = np.asarray(vecs, dtype=np.float32)
        return arr.reshape(-1) if is_single else arr.reshape(len(text), self.dimension)

    def embed_texts(
        self,
        texts: List[str],
        prompt_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a batch of texts into a 2D L2-normalized float32 numpy array of shape (N, D)."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        active_prompt = prompt_name if prompt_name is not None else self.query_prompt_name
        bs = batch_size if batch_size is not None else self.batch_size
        encode_kwargs: Dict[str, Any] = {
            "prompt_name": active_prompt,
            "batch_size": bs,
            "show_progress_bar": show_progress_bar,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.truncate_dim

        vecs = self.model.encode(texts, **encode_kwargs)
        arr = np.asarray(vecs, dtype=np.float32)
        return arr.reshape(len(texts), self.dimension)

    def embed_multimodal(
        self,
        text: str,
        image: Union[str, Path, Image.Image],
    ) -> np.ndarray:
        """
        Encode an interleaved text + image input in a single forward pass using '<|image|>'.
        Enables native single-pass multimodal query composition in EmbeddingGemma 2.
        """
        img_dict = self._prepare_image_input(image)
        formatted_text = text if "<|image|>" in text else f"{text} <|image|>"
        payload = {
            "text": formatted_text,
            "image": img_dict["image"],
        }
        encode_kwargs: Dict[str, Any] = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.truncate_dim
        proc_kwargs = self._get_processing_kwargs()
        if proc_kwargs is not None:
            encode_kwargs["processing_kwargs"] = proc_kwargs

        vec = self.model.encode(payload, **encode_kwargs)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    encode_image = embed_image
    encode_images = embed_images
    encode_text = embed_text
    encode_texts = embed_texts


# ── Factory ────────────────────────────────────────────────────────────────────

# Common presets mapping friendly names to open_clip (model, pretrained) tuples
CLIP_PRESETS = {
    "clip-vit-b-32": ("ViT-B-32", "openai"),
    "clip-vit-b-16": ("ViT-B-16", "openai"),
    "clip-vit-l-14": ("ViT-L-14", "openai"),
    "clip-vit-l-14-336": ("ViT-L-14-336", "openai"),
    "clip-vit-bigg-14": ("ViT-bigG-14", "laion2b_s39b_b160k"),
    "eva-clip-vit-b-16": ("EVA02-B-16", "merged2b_s8b_b131k"),
    "siglip-vit-b-16": ("ViT-B-16-SigLIP", "webli"),
    "siglip-vit-l-16": ("ViT-L-16-SigLIP-256", "webli"),
}


class EmbeddingManager:
    """
    Factory that creates the right embedding backend from a simple name.

    Usage
    -----
    >>> emb = EmbeddingManager.create("clip-vit-b-32")          # CLIP
    >>> emb = EmbeddingManager.create("embeddinggemma-2")       # Gemma 2
    >>> emb = EmbeddingManager.create("vgg19")                  # timm legacy
    >>> emb = EmbeddingManager.create("clip-vit-l-14", device="cuda")
    """

    @staticmethod
    def create(
        model_name: str = "clip-vit-b-32",
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> BaseEmbedding:
        """
        Create an embedding backend.

        Parameters
        ----------
        model_name : str
            A preset name (e.g. 'clip-vit-b-32', 'embeddinggemma-2') or a raw model name.
        device : str or None
            Target device.
        batch_size : int
            Batch size for extraction.

        Returns
        -------
        BaseEmbedding
        """
        name_lower = model_name.lower().strip()

        # Check EmbeddingGemma 2 presets or model names
        if name_lower in GEMMA2_PRESETS or "embeddinggemma-2" in name_lower:
            # Filter out legacy kwargs for timm/clip to prevent TypeError in SentenceTransformer
            kwargs.pop("pretrained", None)
            kwargs.pop("image_size", None)
            explicit_truncate_dim = kwargs.pop("truncate_dim", None)
            if name_lower in GEMMA2_PRESETS:
                preset_cfg = GEMMA2_PRESETS[name_lower]
                preset_dim = preset_cfg["truncate_dim"]
                truncate_dim = explicit_truncate_dim if explicit_truncate_dim is not None else preset_dim
                model_id = kwargs.pop("model_id", preset_cfg["model_id"])
            else:
                truncate_dim = explicit_truncate_dim
                model_id = kwargs.pop("model_id", model_name)

            return EmbeddingGemma2Embedding(
                model_id=model_id,
                device=device,
                truncate_dim=truncate_dim,
                batch_size=batch_size,
                **kwargs,
            )

        # Check CLIP presets
        if name_lower in CLIP_PRESETS:
            clip_model, clip_pretrained = CLIP_PRESETS[name_lower]
            return CLIPEmbedding(
                model_name=clip_model,
                pretrained=clip_pretrained,
                device=device,
                batch_size=batch_size,
            )

        # Check if it looks like an open_clip model name (contains 'ViT' or 'clip' or 'siglip')
        if any(k in name_lower for k in ("vit-", "clip", "siglip", "eva")):
            pretrained = kwargs.get("pretrained", "openai")
            return CLIPEmbedding(
                model_name=model_name,
                pretrained=pretrained,
                device=device,
                batch_size=batch_size,
            )

        # Fall back to timm
        return TimmEmbedding(
            model_name=model_name,
            pretrained=kwargs.get("pretrained", True),
            image_size=kwargs.get("image_size", 224),
            device=device,
            batch_size=batch_size,
        )

    @staticmethod
    def list_presets() -> dict:
        """Return available CLIP preset names and their model details."""
        return dict(CLIP_PRESETS)
