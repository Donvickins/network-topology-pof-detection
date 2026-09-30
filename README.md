# Network Topology Point-of-Failure (POF) Detection System

> ## ⚠️ Disclaimer
>
> This repository is an **internal project** developed during an internship and is
> published here solely as an engineering portfolio piece for recruiters. It is
> **not intended for public or production use** and has never been operated as a
> public service.
>
> All trained models (`models/YOLO/best.pt`, `models/GNN/best.pt`), the custom
> OCR language model (`pof_ocr`), and the proprietary training dataset have been
> **removed** from this repository. The code is shared to demonstrate the
> architecture and engineering approach only.

## Overview

A Python-based computer-vision system that detects and predicts the **point of
failure (POF)** in network topology diagrams. Given an image of a network
topology and the ID of a site that is currently down, the system predicts which
site is the most likely root cause and returns a confidence score.

It combines three models working in a single inference pipeline:

1. **YOLOv8 instance segmentation** - detects network nodes (ATN, RTN, Router,
   Switch, HubSite) and the links (edges) between them, along with their color
   encoding.
2. **Tesseract OCR** with a custom-trained `pof_ocr` language model - reads the
   site IDs stamped next to each detected node.
3. **Graph Convolutional Network (GATv2)** - converts the detected nodes and
   links into a graph and reasons over its structure to localize the point of
   failure, with a dual head that first decides whether a POF exists and then
   which site it is.

The system is exposed as a **FastAPI** service and is also packaged into a
standalone **Windows desktop executable** via Nuitka.

## Highlights

- **End-to-end vision pipeline**: raw topology image in, predicted POF + certainty out.
- **Graph-based reasoning**: failure localization over the topology graph, not
  just per-node classification.
- **Dual-head GNN**: a graph-level head determines if a failure exists, and a
  per-node head localizes where. The model can return `indeterminate` when
  uncertain.
- **Custom OCR**: a purpose-trained Tesseract language model plus HSV color
  masking for reliable site-ID extraction.
- **Fault-tolerant matching**: fuzzy string matching (`rapidfuzz`, threshold 70,
  set in `config.yaml`) aligns user-supplied site IDs to OCR output.
- **Deployable as service or desktop app**: FastAPI + uvicorn, or a packaged
  Windows `.exe`.

## How It Works

```
+------------------+     +----------------------+     +-----------------------+
| Topology image   | --> | YOLOv8 instance      | --> | Graph construction     |
| (base64 upload)  |     | segmentation         |     | (nodes + color-encoded|
+------------------+     | imgsz=1280 conf=0.1  |     |  links)               |
                         +----------------------+     +-----------+-----------+
                                                                    |
                                                                    v
+------------------+     +----------------------+     +-----------------------+
| POF + certainty  | <-- | GATv2 GNN            | <-- | OCR site IDs + fuzzy  |
| (or 'indetermin- |     | dual-head: has_POF   |     | match against down_id|
|  ate')           |     | + per-node POF       |     +-----------------------+
+------------------+     +----------------------+
```

Stage-by-stage:

1. **Detection**: YOLOv8 (`imgsz=1280`, `conf=0.1`) segments network elements.
   Each class is a `{type}_{color}` combination (e.g. `RTN_Yellow`,
   `Link_Green`). Link detections become graph edges; node detections become
   graph nodes.
2. **OCR**: each node's site-ID label is cropped, binarized via HSV color
   masking (upscaled 3×), and read with Tesseract using the custom `pof_ocr`
   language and a `A-Z0-9_` character whitelist.
3. **Matching**: the supplied `site_id` is fuzzy-matched against every
   extracted ID; a match below the 70% threshold is rejected.
4. **Graph construction**: node features are a 12-dim vector
   (5 one-hot type + 6 one-hot color + 1 "is the down site" flag); edges carry
   a 6-dim color encoding. Edges are kept only when both endpoints lie within a
   distance threshold of a node center.
5. **Inference**: a 3-layer **GATv2Conv** network (with LayerNorm + ELU)
   produces node embeddings; a per-node head predicts the POF site, and a
   graph-level head predicts whether a POF exists at all. If the graph-level
   confidence is below 0.5, the system returns `indeterminate`.

## Repository Layout

```
.
├── app.py                     # Entry point: boots the FastAPI/uvicorn server
├── build.py                   # Nuitka Windows standalone build script
├── test.py                    # Manual test script
├── pyproject.toml             # Deps source of truth (uv sync --extra cpu|cuda)
├── uv.lock                    # locked deps
├── config.yaml                # Thresholds, model and server settings
├── Fix.md                     # Code-review risk log
├── core/
│   ├── api.py                 # FastAPI routes (POST /pof, GET /health)
│   ├── demo.py                # Demo mode: hardcoded detections + random GNN
│   │                          # (used only when POF_MODEL_SOURCE=demo)
│   ├── middleware.py          # Request timeout (504)
│   ├── gnn/
│   │   └── model.py           # GATv2 GNN architecture (dual-head)
│   └── utils/
│       ├── auth.py            # Bearer check (POST /pof only)
│       ├── helpers.py         # YOLO result parsing, graph/tensor construction
│       ├── pof.py             # Core business logic: model prep + inference
│       ├── schema.py          # Pydantic request/response models
│       ├── storage.py         # Received-image save (png, uuid, containment)
│       ├── constants.py       # Settings loaded from config.yaml + env
│       ├── paths.py           # Finds the project folder (script or .exe)
│       ├── enums.py           # Fixed error messages for OCR/image failures
│       ├── node_type_config.py# Node-type and color → feature mappings
│       ├── exception_handler.py # Typed domain exceptions
│       └── logger_config.py   # Rotating-file + console logging
├── training/
│   ├── yolo/                  # auto_anotate, bbox_2_seg, merge_segbbox_2_existing_seg,
│   │                          # export_to_cvat, trainer, trainer_colab, pt_to_onnx
│   ├── pof/                   # 01_create_txt_for_images, 02_prep_dataset, 03_gnn_trainer
│   └── ocr/                   # prep_ocr_dataset (+ README)
├── cvat_config/               # classes_raw.txt label configuration
├── models/                    # (stripped) YOLO/ and GNN/ best.pt
├── workspace/                 # classes.txt; received_images/ + samples/ created at runtime
├── logs/                      # app.log + error.log (created at startup)
└── icon/                      # Windows application icon
```

## Tech Stack

| Layer         | Technology                                               |
| ------------- | -------------------------------------------------------- |
| API / Service | Python, FastAPI, uvicorn, fastapi-guard (rate limiting, pen detection, IP banning) |
| Detection     | Ultralytics YOLOv8 (instance segmentation)               |
| OCR           | Tesseract (custom `pof_ocr` model), OpenCV (HSV masking) |
| Graph ML      | PyTorch, PyTorch Geometric (GATv2Conv, LayerNorm)        |
| Matching      | rapidfuzz                                    |
| Packaging     | Nuitka (standalone Windows `.exe`)                       |

## Getting Started

> The trained models and OCR language model are **not** included in this
> repository. Re-train or restore them before the system can produce
> predictions (see the [training pipeline](#training-pipeline)).

```bash
# 1. Install uv (https://docs.astral.sh/uv/), then sync dependencies.
#    CPU is the default path; GPU boxes use --extra cuda instead.
uv sync --extra cpu        # CPU (all systems)
# uv sync --extra cuda     # NVIDIA GPU on Linux/Windows (cu126);
                           # falls back to CPU wheels on macOS

# 2. (Required) Place the trained models at:
#    models/YOLO/best.pt
#    models/GNN/best.pt
#    and make the custom 'pof_ocr' Tesseract model available

# 3. Run the API server (default http://127.0.0.1:5500, see config.yaml)
BEARER_TOKEN=<shared-secret> uv run app.py
```

_The server always starts, even without model files. In live mode it runs
in a limited state: `/health` reports `degraded` and
`/pof` answers `503 Models not available`. Demo mode always reports
`healthy`. All thresholds and model settings live in `config.yaml`._

_Requests are limited to 15 MB bodies (`413`) and images up to 4500px per side
(`400`)._

_`POST /pof` needs `Authorization: Bearer <token>` matching the server's
`BEARER_TOKEN` env var. Missing or wrong token → `401`. If the server itself
has no `BEARER_TOKEN` set, `/pof` answers `503`. `GET /health` needs no
token._

### Try it without models (demo mode)

No weights needed. The server uses hardcoded detections and a random
graph model instead of the real ones, so the answer is an example only,
not a real prediction:

```bash
# 1. Start the server in demo mode
BEARER_TOKEN=demo-token POF_MODEL_SOURCE=demo python app.py

# 2. Check it is running (no token needed)
curl localhost:5500/health
# {"status": "healthy", "environment": "demo"}

# 3. Send the bundled sample image with site_id DEMO1 (token required)
#    (file created on first demo boot at
#    workspace/samples/demo_topology.png)
curl -X POST localhost:5500/pof \
  -H "Authorization: Bearer demo-token" \
  -H "Content-Type: application/json" \
  -d '{"site_id": "DEMO1", "order_id": "DEMO-1", "image_base64": "<base64...>"}'
```

Upload the sample file to `POST /pof` with `"site_id": "DEMO1"`.
You get back a full example response with `pof` set to `DEMO1`.

## Security Notes

`POST /pof` is locked with a bearer token. `GET /health` is open.

- Callers send `Authorization: Bearer <token>` on `POST /pof`. The server
  compares it against its `BEARER_TOKEN` env var with a constant-time
  check. Missing or wrong token → `401`; server has no `BEARER_TOKEN`
  set → `503`.
- Global abuse protection via `fastapi-guard`: 40 requests per 60 seconds,
  penetration-pattern detection, IP banning (10 strikes → 1200 s ban).
- Stability limits: 15 MB body (`413`), 4500px image side (`400`), 25 s
  request timeout (`504`).
- No CORS: no browser calls this app. It is service-to-service behind the
  company firewall, so there is nothing for CORS to allow. The firewall and
  the network team remain the outer layer. It must never face the open
  Internet as it is.

## API Reference

### `POST /pof`

Predicts the point of failure for a topology image. Requires
`Authorization: Bearer <token>` matching the server's `BEARER_TOKEN`.

**Request** — `application/json`:

```json
{
  "site_id": "AKW_023",
  "order_id": "ORD-1042",
  "image_base64": "<base64-encoded topology image>"
}
```

**Response** — `200 OK`:

```json
{
  "status": "success",
  "data": {
    "site_id": "AKW_023",
    "order_id": "ORD-1042",
    "pof": "RTN_045",
    "certainty": 87.34
  }
}
```

`pof` is the predicted site ID, or `indeterminate` when the model's graph-level
confidence is below 0.5. `certainty` is the prediction probability as a
percentage.

### `GET /health`

Shows whether the server can make predictions:

```json
{ "status": "healthy", "environment": "live" }
```

`status` is `healthy` in demo mode. In live mode it is `healthy` when models
are loaded, `degraded` when they are not.
`environment` is `live` or `demo`.

## Training Pipeline

The `training/` directory contains the full model-development pipeline.

### YOLO (`training/yolo/`)

- `auto_anotate.py` - automated annotation of raw topology images.
- `bbox_2_seg.py`, `merge_segbbox_2_existing_seg.py` - segmentation-mask
  conversion and merging.
- `export_to_cvat.py` - export labels to CVAT for review/labelling.
- `trainer.py`, `trainer_colab.py` - YOLOv8 training (local / Google Colab).
- `pt_to_onnx.py` - conversion of trained weights to ONNX.

### GNN (`training/pof/`)

- `01_create_txt_for_images.py`, `02_prep_dataset.py` - build the graph dataset
  from topology images.
- `03_gnn_trainer.py` - trains the GATv2 model, saving `best.pt` / `last.pt`
  under `workspace/trained/run<N>/`.

### OCR (`training/ocr/`)

- `prep_ocr_dataset.py` - prepares training data for the custom Tesseract
  language model (`pof_ocr`).

## Desktop Build (Windows)

`build.py` packages the application into a standalone `POF_Detection.exe`
using Nuitka, bundling the custom Tesseract binary and the Ultralytics config
files so it runs on machines without a Python environment:

```bash
python build.py
```

## Acknowledgements

Developed during an internship at **Huawei Technologies**. This repository is a
stripped-down engineering showcase. All proprietary assets, models, and datasets
have been removed.
