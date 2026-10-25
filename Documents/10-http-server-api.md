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

#### `POST /api/v1/index/delete`

批量从向量索引和元数据目录中删除图像记录。支持通过图像唯一 ID 列表（`image_ids`）或结构化元数据过滤条件（`filters`，支持 `$gte`、`$lte`、`$in`、`$or` 等操作符）指定删除目标。

- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `application/json`

##### Request Body
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_ids` | `array[string]` | Conditional | `null` | 待删除的图像唯一 ID 列表。与 `filters` 至少提供其一。 |
| `filters` | `object` | Conditional | `null` | 结构化元数据过滤条件，匹配该条件的图像均会被删除。与 `image_ids` 至少提供其一。 |
| `save_after_delete` | `boolean` | No | `true` | 删除完成后是否自动将最新索引与元数据落盘保存。 |

##### Request Example (按 ID 列表删除)
```json
{
  "image_ids": [
    "fe4a77edc6599079df4678e011ebaf93",
    "a1b2c3d4e5f6789012345678abcdef01"
  ],
  "save_after_delete": true
}
```

##### Request Example (按结构化条件过滤删除)
```json
{
  "filters": {
    "state": "inactive",
    "duration_sec": {"$lte": 60}
  },
  "save_after_delete": true
}
```

##### Response (200 OK)
```json
{
  "status": "deleted",
  "deleted_count": 2,
  "total_indexed": 1252,
  "deleted_ids": [
    "fe4a77edc6599079df4678e011ebaf93",
    "a1b2c3d4e5f6789012345678abcdef01"
  ]
}
```

##### Error Responses
- `400 Bad Request`: 未提供 `image_ids` 与 `filters`（两者均为空）。
- `401 Unauthorized`: 缺少或无效的 `X-API-Key` 鉴权头。

##### cURL Example
```shell
curl -X POST "http://localhost:8000/api/v1/index/delete" \
  -H "X-API-Key: your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_ids": ["fe4a77edc6599079df4678e011ebaf93"],
    "save_after_delete": true
  }'
```

---

#### `POST /api/v1/index/clear` / `DELETE /api/v1/index`

清空当前索引库中的全部向量与元数据记录（重置为空索引）。同时支持 `POST /api/v1/index/clear` 与 RESTful 风格的 `DELETE /api/v1/index`。

- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `application/json`

##### Request Body (Optional)
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `save_after_clear` | `boolean` | No | `true` | 清空后是否立即将空索引状态持久化写入磁盘。 |

##### Response (200 OK)
```json
{
  "status": "cleared",
  "deleted_count": 1252,
  "total_indexed": 0
}
```

##### Error Responses
- `401 Unauthorized`: 缺少或无效的 `X-API-Key` 鉴权头。

##### cURL Examples
```shell
# 方式一：POST /api/v1/index/clear
curl -X POST "http://localhost:8000/api/v1/index/clear" \
  -H "X-API-Key: your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{"save_after_clear": true}'

# 方式二：DELETE /api/v1/index
curl -X DELETE "http://localhost:8000/api/v1/index" \
  -H "X-API-Key: your-secure-api-key"
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
| `filters` | `object` | No | `null` | Optional metadata filter conditions. Supports equality, range (`$gte`, `$lte`, `$gt`, `$lt`), membership (`$in`, `$nin`), inequality (`$ne`), and logical operators (`$or`, `$and`). |

##### Request Example
```json
{
  "query": "a red vintage sports car parked near lake",
  "k": 5,
  "filters": {
    "category": "vehicles",
    "duration_sec": {"$gte": 1200},
    "ship_size": {"$in": ["small", "medium"]},
    "$or": [{"ship_color": "red"}, {"color_confidence": "low"}]
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

#### `POST /api/v1/search/dual`

双路索引融合搜索接口（Dual-Branch Index Fusion Search）。接收用户输入的搜索词（或语音转写文字）以及可选的查询图片，在服务端直接执行：**结构化过滤优先 $\rightarrow$ 船只 Crop 索引检索得分 $s_{\text{ship}}$ 与整帧/上下文场景索引检索得分 $s_{\text{scene}}$ $\rightarrow$ 按 `group_id` / `track_id` 对齐加权融合（$\text{score} = w_1 \cdot s_{\text{ship}} + w_2 \cdot s_{\text{scene}}$）**，最终返回融合打分排序后的 Top-K 船只目标。

- **Endpoint**: `POST /api/v1/search/dual`
- **Auth**: Required (`X-API-Key`)
- **Content-Type**: `multipart/form-data`

##### Request Form Fields
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | Conditional | `null` | 搜索文字（用户直接输入的文字，或经语音识别转写后的查询文字，如 `"夜间停靠在码头边的红色小型渔船"`）。 |
| `ship_query` | `string` | No | `null` | 可选的独立船只描述词（若不传，服务端直接用 `query` 检索 `ship_crop` 索引）。 |
| `scene_query` | `string` | No | `null` | 可选的独立场景描述词（若不传，服务端直接用 `query` 检索 `full_frame` 索引）。 |
| `file` | `file` (binary) | Conditional | `null` | 以图搜图时上传的查询图片（与 `query` 二选一或同时提供进行图文多模态检索）。 |
| `filters_json` | `string` | No | `null` | 结构化优先过滤条件 JSON 字符串（支持 `camera_location`, `ship_color`, `ship_size`, `state`, `duration_sec` 等过滤，支持 `$gte`, `$lte`, `$in`, `$or`）。 |
| `w1` | `float` | No | `0.7` | 船只 Crop 索引得分权重 $w_1$。 |
| `w2` | `float` | No | `0.3` | 整帧/上下文场景索引得分权重 $w_2$。 |
| `night_color_tolerance` | `boolean` | No | `true` | 是否开启夜间颜色降权容错（开启后对 `color_confidence="low"` 的夜间红外记录不因颜色词不匹配而淘汰）。 |
| `k` | `integer` | No | `20` | 返回对齐去重后的 Top-K 船只结果数（范围：`1` 至 `200`）。 |

> **注意**：`query`、`ship_query`、`scene_query` 与 `file` 至少需提供其一。上传图片在独立临时文件中处理，并在请求结束时由 `finally` 块确保立即清理。

##### Response (200 OK)
```json
{
  "query_mode": "dual_fusion",
  "count": 1,
  "results": [
    {
      "group_id": "GRP_20261022_001",
      "score": 0.918,
      "score_detail": {
        "s_ship": 0.942,
        "s_scene": 0.862
      },
      "ship_crop_image_id": "fe4a77edc6599079df4678e011ebaf93",
      "context_crop_image_id": "77bc88edc6599079df4678e011eb0012",
      "full_frame_image_id": "a1b2c3d4e5f6789012345678abcdef01",
      "metadata": {
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
        "duration_sec": 2100
      }
    }
  ]
}
```

##### Error Responses
- `400 Bad Request`: 未提供任何查询条件（`query`、`ship_query`、`scene_query`、`file` 均为空）、非法的 `filters_json` 格式，或上传了不支持的图片文件类型。
- `401 Unauthorized`: 缺失或无效的 `X-API-Key` 鉴权头。

##### cURL Examples

**Option A: 文本语音转写词融合检索**
```shell
curl -X POST "http://localhost:8000/api/v1/search/dual" \
  -H "X-API-Key: your-secure-api-key" \
  -F "query=夜间停靠在码头边的红色小型渔船" \
  -F "w1=0.7" \
  -F "w2=0.3" \
  -F "filters_json={\"duration_sec\": {\"$gte\": 1200}, \"ship_size\": {\"$in\": [\"small\", \"medium\"]}}" \
  -F "night_color_tolerance=true" \
  -F "k=20"
```

**Option B: 独立船只描述与场景描述检索**
```shell
curl -X POST "http://localhost:8000/api/v1/search/dual" \
  -H "X-API-Key: your-secure-api-key" \
  -F "ship_query=红色小型渔船" \
  -F "scene_query=夜间码头货运区泊位" \
  -F "w1=0.7" \
  -F "w2=0.3" \
  -F "k=10"
```

**Option C: 图片 + 结构化属性过滤融合检索**
```shell
curl -X POST "http://localhost:8000/api/v1/search/dual" \
  -H "X-API-Key: your-secure-api-key" \
  -F "file=@target_ship_reference.jpg" \
  -F "scene_query=集装箱码头泊位" \
  -F "filters_json={\"state\": \"stationary\"}" \
  -F "k=10"
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

#### `DELETE /api/v1/records/{image_id}`

从向量索引和元数据目录中删除指定的单张图像记录。

- **Auth**: Required (`X-API-Key`)

##### Path Parameters
| Parameter | Type | Required | Description |
|---|---|---|---|
| `image_id` | `string` | **Yes** | 待删除的图像唯一标识符。 |

##### Query Parameters
| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `save_after_delete` | `boolean` | No | `true` | 删除完成后是否自动持久化写入磁盘。 |

##### Response (200 OK)
```json
{
  "status": "deleted",
  "image_id": "fe4a77edc6599079df4678e011ebaf93",
  "deleted_count": 1,
  "total_indexed": 1251
}
```

##### Error Responses
- `404 Not Found`: 指定的 `image_id` 在索引与元数据库中均不存在。
- `401 Unauthorized`: 缺少或无效的 `X-API-Key` 鉴权头。

##### cURL Example
```shell
curl -X DELETE "http://localhost:8000/api/v1/records/fe4a77edc6599079df4678e011ebaf93?save_after_delete=true" \
  -H "X-API-Key: your-secure-api-key"
```

---

## 5. 高级扩展接口设计规范与多模态双路融合搜索

### 5.1 扩展 `filters` 支持范围与逻辑或操作符

现有单向量检索及等值匹配（Equality Filter）已全面扩展。在 `/api/v1/search/*`（包括 `/search/text`、`/search/image`、`/search/hybrid` 和 `/search/dual`）的 `filters` / `filters_json` 中，原生支持以下结构化过滤操作符：

#### 5.1.1 操作符规范与语义

1. **`$gte` / `$lte`（数值与时间区间过滤）**：
   - 过滤最小/最大持续时间（`duration_sec`）：`{"duration_sec": {"$gte": 1200, "$lte": 7200}}`；
   - 过滤发生时间区间（`start_time`，支持 ISO 格式与标准时间字符串）：`{"start_time": {"$gte": "2026-10-22 20:00:00", "$lte": "2026-10-22 23:59:59"}}`。
   - 同时也支持严格大于 `$gt` 与严格小于 `$lt`。

2. **`$in` / `$nin`（集合多选与排除过滤）**：
   - 多选船型：`{"ship_type": {"$in": ["fishing_boat", "cargo_ship"]}}`；
   - 多选尺寸：`{"ship_size": {"$in": ["small", "medium"]}}`；
   - 多选监控区域或泊位地点：`{"camera_location": {"$in": ["东港区3号货运码头", "西港区1号泊位"]}}`；
   - 排除特定目标：`{"ship_type": {"$nin": ["tugboat"]}}`。

3. **`$or` / `$and`（逻辑组合过滤）**：
   - 满足任意分支条件：`{"$or": [{"ship_color": "red"}, {"color_confidence": "low"}]}`；
   - 满足所有复合分支条件：`{"$and": [{"state": "stationary"}, {"$or": [{"ship_color": "red"}, {"color_confidence": "low"}]}]}`。

4. **`$eq` / `$ne`（精确匹配与不等值过滤）**：
   - 过滤非移动状态：`{"state": {"$ne": "moving"}}`。

#### 5.1.2 组合过滤示例
```json
{
  "duration_sec": {"$gte": 1200},
  "ship_size": {"$in": ["small", "medium"]},
  "$or": [
    {"ship_color": "red"},
    {"color_confidence": "low"}
  ]
}
```

---

### 5.2 新增双路索引融合搜索接口：`POST /api/v1/search/dual`

#### 5.2.1 业务背景与设计目标
在港口监控与船舶布控场景下，单一的全局大图检索容易丢失船体局部细节（如船舷颜色、特定涂装、吃水标记），而单一的船只 Crop 切片检索则缺乏泊位、码头、岸线等宏观上下文。

为此，系统采用**“双路索引拆开检索再按 Track/Group 对齐加权融合”**的设计：
1. **结构化过滤优先**：先根据发生时间、持续时长、泊位区域及夜间颜色容错条件进行精准剪枝；
2. **双路并行召回**：
   - **船只局部路（Ship Crop Branch）**：检索 `ship_crop` 切片向量索引，得到针对船只细节的相似度得分 $s_{\text{ship}}$；
   - **场景/上下文路（Scene / Context Branch）**：检索 `full_frame` 或 `context_crop` 向量索引，得到针对泊位与宏观场景的相似度得分 $s_{\text{scene}}$；
3. **按 `group_id` / `track_id` 对齐融合**：
   $$\text{score} = w_1 \cdot s_{\text{ship}} + w_2 \cdot s_{\text{scene}}$$
   默认权重 $w_1 = 0.7$（偏重船只特征），$w_2 = 0.3$（场景环境辅助判定）；
4. **输出对齐后的目标组结构**：返回统一的 `group_id`、融合分 `score`、分项得分明细 `score_detail`、各路图像 ID（`ship_crop_image_id`, `context_crop_image_id`, `full_frame_image_id`）及属性元数据。

#### 5.2.2 夜间低照度颜色容错机制（`night_color_tolerance`）
当参数 `night_color_tolerance=true`（默认开启）时：
- 服务端检测到过滤条件中包含 `ship_color`（如 `{"ship_color": "red"}`）；
- 自动将其转换为宽松的 OR 条件：
  ```json
  {"$or": [{"ship_color": "red"}, {"color_confidence": "low"}]}
  ```
- 从而确保在夜间低照度红外模式下，因缺少光照导致颜色置信度判定为 `"low"` 的有效目标不会被颜色词过滤错误淘汰。

---

## 6. End-to-End Python Client Examples

### 6.1 Using `httpx` (Text, Image & Dual Fusion Search)

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

# 3. Standard text search
search_resp = client.post(
    "/api/v1/search/text",
    json={"query": "outdoor scenic view", "k": 3},
)
results = search_resp.json()["results"]
for r in results:
    print(f"Match ID: {r['id']}, Score: {r['score']:.4f}")

# 4. Dual-branch fusion search (/api/v1/search/dual)
dual_resp = client.post(
    "/api/v1/search/dual",
    data={
        "query": "夜间停靠在码头边的红色小型渔船",
        "ship_query": "红色小型渔船",
        "scene_query": "夜间货运码头泊位",
        "w1": 0.7,
        "w2": 0.3,
        "night_color_tolerance": "true",
        "filters_json": '{"duration_sec": {"$gte": 1200}, "ship_size": {"$in": ["small", "medium"]}}',
        "k": 10,
    },
)
dual_data = dual_resp.json()
print(f"Dual fusion returned {dual_data['count']} groups:")
for item in dual_data["results"]:
    print(f"  Group ID: {item['group_id']}")
    print(f"  Fused Score: {item['score']} (Ship: {item['score_detail']['s_ship']}, Scene: {item['score_detail']['s_scene']})")
    print(f"  Ship Crop ID: {item['ship_crop_image_id']}, Full Frame ID: {item['full_frame_image_id']}")
    print(f"  Metadata: {item['metadata']}")

# 5. Delete single record by ID
del_resp = client.delete(f"/api/v1/records/{results[0]['id']}?save_after_delete=true")
print("Deleted single record:", del_resp.json())

# 6. Batch delete by metadata filter
batch_del = client.post(
    "/api/v1/index/delete",
    json={"filters": {"tag": "outdoor"}},
)
print("Batch deleted by filter:", batch_del.json())

# 7. Clear entire index
clear_resp = client.post("/api/v1/index/clear", json={"save_after_clear": True})
print("Cleared index:", clear_resp.json())
```

### 6.2 Using `requests`

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

## 7. Production Deployment Recommendations

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
