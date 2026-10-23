# HTTP Server REST API Reference

DeepImageSearch provides a production-grade, high-performance FastAPI REST server module (`DeepImageSearch.server`) and a dedicated CLI entrypoint (`deep-image-search-server`). It exposes all core capabilities—vector indexing, batch uploads, semantic text search, visual similarity search, hybrid search, metadata management, and binary image streaming—to external microservices, web apps, mobile clients, and cross-language frontends.

---

## 1. Architecture & Key Features

- **Asynchronous Execution & Non-blocking I/O**: Heavy GPU/CPU operations (neural network forward passes, FAISS index updates, disk writes) run in thread pools via FastAPI, keeping the Uvicorn event loop responsive.
- **Concurrency & Thread Safety**: All index mutation operations (`/index/*`) are synchronized via a re-entrant mutual exclusion lock (`threading.RLock`) to prevent index corruption under concurrent writes.
- **Leak-Free Image Handling**:
  - Search query uploads are processed in isolated temporary files and guaranteed to be deleted in `finally` blocks.
  - Ingestion uploads are content-addressed using SHA-256 digests and safely persisted under the configured `upload_dir`.
- **Flexible Authentication**: Optional API Key security (`X-API-Key` header) enabled automatically whenever `DIS_API_KEY` or `--api-key` is configured.
- **Interactive OpenAPI Documentation**: Built-in Swagger UI at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

---

## 2. Installation & Quick Start

### Installation

Install the package with the `[api]` optional dependency group:

```shell
pip install "DeepImageSearch[api]"
```

Or with `uv`:

```shell
uv pip install "DeepImageSearch[api]"
```

### Starting the Server

```shell
deep-image-search-server \
  --model clip-vit-b-32 \
  --index-path ./dis_index \
  --upload-dir ./dis_uploads \
  --port 8000
```

To enable API Key authentication:

```shell
deep-image-search-server --api-key "your-secure-api-key"
```

### CLI Configuration Flags & Environment Variables

| Flag | Environment Variable | Type | Default | Description |
|---|---|---|---|---|
| `--host` | `DIS_HOST` | `str` | `0.0.0.0` | Bind host address. |
| `--port` | `DIS_PORT` | `int` | `8000` | Bind port number. |
| `--model` | `DIS_MODEL_NAME` | `str` | `clip-vit-b-32` | Embedding model preset (`clip-vit-b-32`, `embeddinggemma-2`, etc.) or local weight path. |
| `--vector-store` | `DIS_VECTOR_STORE` | `str` | `faiss` | Vector store backend (`faiss`, `chroma`, `qdrant`). |
| `--index-path` | `DIS_INDEX_PATH` | `str` | `./dis_index` | Persistence directory for vector index files and image metadata records. |
| `--upload-dir` | `DIS_UPLOAD_DIR` | `str` | `./dis_uploads` | Managed filesystem directory where batch-uploaded images are permanently stored. |
| `--device` | `DIS_DEVICE` | `str` | `None` | Compute device (`cuda`, `mps`, `cpu`). Defaults to auto-detect. |
| `--api-key` | `DIS_API_KEY` | `str` | `None` | Authentication secret. If set, clients must send `X-API-Key: <key>`. |
| `--reload` | - | `bool` | `False` | Enable auto-reload for local development. |

---

## 3. Authentication & Common Headers

### Authentication Header

When `--api-key` or `DIS_API_KEY` is set, all endpoints under `/api/v1/*` require the `X-API-Key` HTTP header:

```http
X-API-Key: your-secure-api-key
```

If the header is missing or incorrect, the server responds with:
- **Status Code**: `401 Unauthorized`
- **Body**: `{"detail": "Invalid or missing X-API-Key header."}`

The `/health` endpoint is **public** and does not require authentication.

### Supported Image Formats

Endpoints accepting image files enforce an extension whitelist:
`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, `.tiff`.

---

## 4. API Endpoints Specification

### 4.1 System & Health Endpoints

---

#### `GET /health`

Lightweight health and readiness probe for load balancers (Kubernetes, AWS ALB, NGINX).

- **Auth**: Public (No authentication required)
- **Parameters**: None

##### Response (200 OK)
```json
{
  "status": "ok",
  "model_name": "clip-vit-b-32",
  "vector_store": "faiss",
  "indexed_count": 1250
}
```

##### Field Descriptions
| Field | Type | Description |
|---|---|---|
| `status` | `string` | Service status (always `"ok"` when running). |
| `model_name` | `string` | Currently loaded embedding model preset or path. |
| `vector_store` | `string` | Active vector store backend. |
| `indexed_count` | `integer` | Total number of images currently indexed. |

##### cURL Example
```shell
curl -X GET "http://localhost:8000/health"
```

---

#### `GET /api/v1/info`

Retrieve detailed summary of the running engine, vector store configuration, and storage locations.

- **Auth**: Required (`X-API-Key`)
- **Parameters**: None

##### Response (200 OK)
```json
{
  "model": "clip-vit-b-32",
  "dimension": 512,
  "indexed_images": 1250,
  "supports_text_search": true,
  "vector_store": "FAISSStore",
  "metadata_store": "JsonMetadataStore",
  "index_dir": "./dis_index"
}
```

##### cURL Example
```shell
curl -X GET "http://localhost:8000/api/v1/info" \
  -H "X-API-Key: your-secure-api-key"
```

---

### 4.2 Indexing Endpoints

---

#### `POST /api/v1/index/paths`

Index images directly from paths or folders accessible on the server filesystem.

- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `application/json`

##### Request Body
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `paths` | `string` or `array[string]` | **Yes** | - | Directory path (e.g. `"/data/photos"`) or list of specific image file paths. |
| `generate_captions` | `boolean` | No | `false` | Whether to auto-generate image captions using the configured LLM captioner. |
| `metadata` | `object` or `array[object]` | No | `null` | Custom metadata dictionary or list of dictionaries corresponding to `paths`. |
| `save_after_index` | `boolean` | No | `true` | Automatically persist updated vector index and metadata to disk upon completion. |

##### Request Example
```json
{
  "paths": ["/data/photos/nature1.jpg", "/data/photos/nature2.jpg"],
  "generate_captions": false,
  "metadata": [
    {"category": "landscape", "author": "Alice"},
    {"category": "wildlife", "author": "Bob"}
  ],
  "save_after_index": true
}
```

##### Response (200 OK)
```json
{
  "status": "indexed",
  "added_count": 2,
  "total_indexed": 1252,
  "index_path": "./dis_index"
}
```

##### Error Responses
- `400 Bad Request`: Invalid paths or failure during feature extraction.
- `401 Unauthorized`: Missing or invalid API key.

##### cURL Example
```shell
curl -X POST "http://localhost:8000/api/v1/index/paths" \
  -H "X-API-Key: your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "paths": "/data/photos",
    "generate_captions": false,
    "save_after_index": true
  }'
```

---

#### `POST /api/v1/index/upload`

Upload one or more image files over HTTP from external clients (browser, mobile app, microservice). The server automatically stores the files in `upload_dir` with content-addressed SHA-256 filenames, extracts embeddings, registers metadata, and updates the index.

- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `multipart/form-data`

##### Form Fields
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `files` | `file` (binary) | **Yes** | - | One or more binary image files to upload (repeatable field). |
| `generate_captions` | `boolean` | No | `false` | Auto-generate captions with configured vision LLM. |
| `metadata_json` | `string` | No | `null` | Optional JSON string containing a metadata object or array of metadata objects matching the uploaded files. |
| `save_after_index` | `boolean` | No | `true` | Persist index to disk immediately after insertion. |

##### Response (200 OK)
```json
{
  "status": "indexed",
  "added_count": 2,
  "total_indexed": 1254,
  "index_path": "./dis_index"
}
```

##### Error Responses
- `400 Bad Request`: No files provided, empty file contents, invalid `metadata_json`, or unsupported file format.
- `401 Unauthorized`: Missing or invalid API key.

##### cURL Example
```shell
curl -X POST "http://localhost:8000/api/v1/index/upload" \
  -H "X-API-Key: your-secure-api-key" \
  -F "files=@photo_a.jpg" \
  -F "files=@photo_b.png" \
  -F "metadata_json={\"uploaded_by\": \"admin\", \"batch_id\": 42}" \
  -F "save_after_index=true"
```

---

#### `POST /api/v1/index/save`

Explicitly persist current in-memory vector index and metadata records to disk.

- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `application/json`

##### Request Body (Optional)
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `index_path` | `string` | No | `null` | Target directory override. If omitted, uses the configured default `index_path`. |

##### Request Example
```json
{
  "index_path": "./backups/index_20261023"
}
```

##### Response (200 OK)
```json
{
  "status": "saved",
  "index_path": "./backups/index_20261023",
  "total_indexed": 1254
}
```

##### cURL Example
```shell
curl -X POST "http://localhost:8000/api/v1/index/save" \
  -H "X-API-Key: your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

#### `POST /api/v1/index/load`

Reload the vector store index and image records from disk.

- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `application/json`

##### Request Body (Optional)
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `index_path` | `string` | No | `null` | Directory path containing saved index files. Defaults to configured `index_path`. |

##### Response (200 OK)
```json
{
  "status": "loaded",
  "index_path": "./dis_index",
  "total_indexed": 1254
}
```

##### Error Responses
- `404 Not Found`: Target directory does not exist or index files cannot be found.
- `401 Unauthorized`: Missing or invalid API key.

##### cURL Example
```shell
curl -X POST "http://localhost:8000/api/v1/index/load" \
  -H "X-API-Key: your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

### 4.3 Search Endpoints

---

#### `POST /api/v1/search/text`

Semantic text-to-image search. Encodes a natural language text query into vector space and retrieves the nearest indexed images.

- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `application/json`

##### Request Body
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | **Yes** | - | Natural language text query (minimum length 1). |
| `k` | `integer` | No | `10` | Number of nearest results to return (range: `1` to `200`). |
| `filters` | `object` | No | `null` | Optional key-value equality filter conditions applied against metadata attributes. |

##### Request Example
```json
{
  "query": "a red vintage sports car parked near lake",
  "k": 5,
  "filters": {
    "category": "vehicles"
  }
}
```

##### Response (200 OK)
```json
{
  "query_mode": "text",
  "count": 2,
  "results": [
    {
      "id": "fe4a77edc6599079df4678e011ebaf93",
      "score": 0.8942,
      "metadata": {
        "image_id": "fe4a77edc6599079df4678e011ebaf93",
        "image_index": 14,
        "image_name": "red_car_lake.jpg",
        "image_path": "/dataset/photos/red_car_lake.jpg",
        "caption": "A classic red sports coupe parked beside a mountain lake",
        "category": "vehicles",
        "indexed_at": "2026-10-23T12:00:00+00:00"
      }
    },
    {
      "id": "a1b2c3d4e5f6789012345678abcdef01",
      "score": 0.7415,
      "metadata": {
        "image_id": "a1b2c3d4e5f6789012345678abcdef01",
        "image_index": 28,
        "image_name": "vintage_red_auto.jpg",
        "image_path": "/dataset/photos/vintage_red_auto.jpg",
        "category": "vehicles",
        "indexed_at": "2026-10-23T12:05:00+00:00"
      }
    }
  ]
}
```

##### Error Responses
- `400 Bad Request`: Model does not support text search (e.g. image-only timm CNN models) or invalid query.
- `401 Unauthorized`: Missing or invalid API key.

##### cURL Example
```shell
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "X-API-Key: your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "golden sunset over mountains",
    "k": 5
  }'
```

---

#### `POST /api/v1/search/image`

Visual similarity image-to-image search. Finds images visually similar to an uploaded query image or a server-side image path.

- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `multipart/form-data`

##### Form Fields
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | `file` (binary) | Conditional | `null` | Uploaded query image file. Required if `image_path` is not provided. |
| `image_path` | `string` | Conditional | `null` | Path to an existing image on the server. Required if `file` is not uploaded. |
| `k` | `integer` | No | `10` | Number of results to return (`1` to `200`). |
| `filters_json` | `string` | No | `null` | JSON string representing metadata filter conditions (e.g. `'{"category": "shoes"}'`). |

> **Note**: Exactly one of `file` or `image_path` must be supplied. If an image file is uploaded, it is automatically processed in a temporary file and removed immediately after embedding extraction.

##### Response (200 OK)
```json
{
  "query_mode": "image",
  "count": 1,
  "results": [
    {
      "id": "9876543210abcdef0123456789abcdef",
      "score": 0.9651,
      "metadata": {
        "image_id": "9876543210abcdef0123456789abcdef",
        "image_index": 3,
        "image_name": "sample_shoe.jpg",
        "image_path": "/dataset/photos/sample_shoe.jpg",
        "indexed_at": "2026-10-23T12:00:00+00:00"
      }
    }
  ]
}
```

##### Error Responses
- `400 Bad Request`: Neither `file` nor `image_path` provided, unsupported file extension, or invalid `filters_json`.
- `404 Not Found`: Given `image_path` does not exist on the server.
- `401 Unauthorized`: Missing or invalid API key.

##### cURL Examples
**Option A: Upload local image query**
```shell
curl -X POST "http://localhost:8000/api/v1/search/image" \
  -H "X-API-Key: your-secure-api-key" \
  -F "file=@query_shoe.jpg" \
  -F "k=5"
```

**Option B: Use server-side path**
```shell
curl -X POST "http://localhost:8000/api/v1/search/image" \
  -H "X-API-Key: your-secure-api-key" \
  -F "image_path=/dataset/photos/reference.jpg" \
  -F "k=5"
```

---

#### `POST /api/v1/search/hybrid`

Multi-modal hybrid query combining text intent and visual reference. Supports weighted vector fusion and native interleaved multimodal queries (EmbeddingGemma 2).

- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `multipart/form-data`

##### Form Fields
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text_query` | `string` | **Yes** | - | Natural language text description. |
| `file` | `file` (binary) | Conditional | `null` | Uploaded reference image file (required if `image_path` not provided). |
| `image_path` | `string` | Conditional | `null` | Server-side reference image file path. |
| `text_weight` | `float` | No | `0.5` | Weight for text query vs image query between `0.0` and `1.0`. Image weight is `(1.0 - text_weight)`. |
| `mode` | `string` | No | `"hybrid"` | `"hybrid"` for weighted vector fusion, or `"interleaved"` for EmbeddingGemma 2 native `<|image|>` single-pass. |
| `k` | `integer` | No | `10` | Number of results to return (`1` to `200`). |
| `filters_json` | `string` | No | `null` | JSON string of metadata filter conditions. |

##### Response (200 OK)
```json
{
  "query_mode": "hybrid",
  "count": 1,
  "results": [
    {
      "id": "e4d3c2b1a0987654321fedcba0123456",
      "score": 0.8876,
      "metadata": {
        "image_id": "e4d3c2b1a0987654321fedcba0123456",
        "image_name": "trail_sneaker_blue.jpg",
        "image_path": "/dataset/photos/trail_sneaker_blue.jpg",
        "caption": "Blue waterproof trail running shoe with rugged sole"
      }
    }
  ]
}
```

##### cURL Example
```shell
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: your-secure-api-key" \
  -F "text_query=waterproof trail edition" \
  -F "file=@hiking_shoe.jpg" \
  -F "text_weight=0.6" \
  -F "mode=hybrid" \
  -F "k=5"
```

---

### 4.4 Records & Media Streaming Endpoints

---

#### `GET /api/v1/records`

Retrieve a paginated list of indexed image metadata records stored in the metadata catalog.

- **Auth**: Required (`X-API-Key`)

##### Query Parameters
| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `offset` | `integer` | No | `0` | Number of records to skip (must be `>= 0`). |
| `limit` | `integer` | No | `50` | Maximum number of records to return (range: `1` to `500`). |

##### Response (200 OK)
```json
{
  "total": 1254,
  "offset": 0,
  "limit": 2,
  "records": [
    {
      "image_id": "fe4a77edc6599079df4678e011ebaf93",
      "image_index": 0,
      "image_name": "red_car_lake.jpg",
      "image_path": "/dataset/photos/red_car_lake.jpg",
      "caption": "A classic red sports coupe parked beside a mountain lake",
      "indexed_at": "2026-10-23T12:00:00+00:00"
    },
    {
      "image_id": "9876543210abcdef0123456789abcdef",
      "image_index": 1,
      "image_name": "sample_shoe.jpg",
      "image_path": "/dataset/photos/sample_shoe.jpg",
      "indexed_at": "2026-10-23T12:00:00+00:00"
    }
  ]
}
```

##### cURL Example
```shell
curl -X GET "http://localhost:8000/api/v1/records?offset=0&limit=50" \
  -H "X-API-Key: your-secure-api-key"
```

---

#### `GET /api/v1/records/{image_id}`

Retrieve the complete metadata record for a specific image by its unique ID.

- **Auth**: Required (`X-API-Key`)

##### Path Parameters
| Parameter | Type | Required | Description |
|---|---|---|---|
| `image_id` | `string` | **Yes** | Unique image identifier (e.g. MD5 digest or custom ID). |

##### Response (200 OK)
```json
{
  "image_id": "fe4a77edc6599079df4678e011ebaf93",
  "image_index": 0,
  "image_name": "red_car_lake.jpg",
  "image_path": "/dataset/photos/red_car_lake.jpg",
  "caption": "A classic red sports coupe parked beside a mountain lake",
  "category": "vehicles",
  "indexed_at": "2026-10-23T12:00:00+00:00"
}
```

##### Error Responses
- `404 Not Found`: No record found matching the specified `image_id`.

##### cURL Example
```shell
curl -X GET "http://localhost:8000/api/v1/records/fe4a77edc6599079df4678e011ebaf93" \
  -H "X-API-Key: your-secure-api-key"
```

---

#### `GET /api/v1/records/{image_id}/content`

Stream the original binary image file directly over HTTP. This allows external web frontends, mobile apps, or HTML `<img>` tags to display search results directly without separate static web hosting.

- **Auth**: Required (`X-API-Key`)
- **Response Content-Type**: `image/jpeg`, `image/png`, `image/webp`, etc. (inferred from file)

##### Path Parameters
| Parameter | Type | Required | Description |
|---|---|---|---|
| `image_id` | `string` | **Yes** | Unique image identifier. |

##### Response (200 OK)
Returns binary image stream with standard media headers:
```http
HTTP/1.1 200 OK
content-type: image/jpeg
content-length: 184320
```

##### Error Responses
- `404 Not Found`: Record not found or the underlying file is missing from disk.

##### cURL Example
```shell
# Download and save the image content
curl -X GET "http://localhost:8000/api/v1/records/fe4a77edc6599079df4678e011ebaf93/content" \
  -H "X-API-Key: your-secure-api-key" \
  -o result.jpg
```

---

## 5. End-to-End Python Client Examples

### Using `httpx`

```python
import httpx

BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "your-secure-api-key"}

client = httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=30.0)

# 1. Health check
health = client.get("/health").json()
print("Server status:", health["status"], "Total indexed:", health["indexed_count"])

# 2. Upload and index images
with open("local_photo.jpg", "rb") as f:
    upload_resp = client.post(
        "/api/v1/index/upload",
        files=[("files", ("local_photo.jpg", f, "image/jpeg"))],
        data={"metadata_json": '{"tag": "outdoor"}'},
    )
print("Upload status:", upload_resp.json())

# 3. Text search
search_resp = client.post(
    "/api/v1/search/text",
    json={"query": "outdoor scenic view", "k": 3},
)
results = search_resp.json()["results"]
for r in results:
    print(f"Match ID: {r['id']}, Score: {r['score']:.4f}")

# 4. Stream and display first image result
if results:
    top_id = results[0]["id"]
    content_resp = client.get(f"/api/v1/records/{top_id}/content")
    with open("downloaded_top_match.jpg", "wb") as f:
        f.write(content_resp.content)
    print("Saved top match image locally.")
```

### Using `requests`

```python
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "your-secure-api-key"}

# Visual similarity search with a query image
with open("query_sample.jpg", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/api/v1/search/image",
        headers=HEADERS,
        files={"file": ("query.jpg", f, "image/jpeg")},
        data={"k": 5},
    )

print("Similar images found:", resp.json())
```

---

## 6. Production Deployment Recommendations

### Running with Gunicorn / Uvicorn Workers

In production environments requiring multi-worker parallelism on a multi-GPU machine or read-heavy setup:

```shell
uvicorn DeepImageSearch.server.app:create_app \
  --factory \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

> [!TIP]
> When scaling out multiple Uvicorn workers with concurrent writes (`/index/*`), use a shared network filesystem or vector store backend like **Qdrant** or **PostgreSQL** (`psycopg2`) to maintain consistent state across worker processes. For single-node setups using FAISS, the default 1-worker mode with internal thread-pooling provides maximum throughput with zero synchronization overhead.
