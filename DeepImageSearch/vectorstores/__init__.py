# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
from DeepImageSearch.vectorstores.base import BaseVectorStore
from DeepImageSearch.vectorstores.faiss_store import FAISSStore

__all__ = ["BaseVectorStore", "FAISSStore"]

# Optional imports
try:
    import chromadb  # noqa: F401
    from DeepImageSearch.vectorstores.chroma_store import ChromaStore
    __all__.append("ChromaStore")
except ImportError:
    pass

try:
    import qdrant_client  # noqa: F401
    from DeepImageSearch.vectorstores.qdrant_store import QdrantStore
    __all__.append("QdrantStore")
except ImportError:
    pass
