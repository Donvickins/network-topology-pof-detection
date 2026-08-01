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
- **Fault-tolerant matching**: fuzzy string matching (`fuzzywuzzy`, threshold 70)
  aligns user-supplied site IDs to OCR output.
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
├── requirements.txt
├── core/
│   ├── api.py                 # FastAPI routes (POST /pof, GET /health)
│   ├── gnn/
│   │   └── model.py           # GATv2 GNN architecture (dual-head)
│   └── utils/
│       ├── helpers.py         # YOLO result parsing, graph/tensor construction
│       ├── pof.py             # Core business logic: model prep + inference
│       ├── schema.py          # Pydantic request/response models
│       ├── node_type_config.py# Node-type and color → feature mappings
│       ├── exception_handler.py
│       └── logger_config.py   # Rotating-file + console logging
├── training/
│   ├── yolo/                  # YOLO dataset prep, CVAT export, trainers, ONNX
│   ├── pof/                   # GNN dataset prep + graph trainer
│   └── ocr/                   # OCR dataset prep
├── cvat_config/               # CVAT label configuration
├── models/                    # (stripped) best.pt for YOLO and GNN
├── workspace/                 # classes.txt; received_images/ created at runtime
└── icon/                      # Windows application icon
```

## Tech Stack

| Layer         | Technology                                               |
| ------------- | -------------------------------------------------------- |
| API / Service | Python, FastAPI, uvicorn                                 |
| Detection     | Ultralytics YOLOv8 (instance segmentation)               |
| OCR           | Tesseract (custom `pof_ocr` model), OpenCV (HSV masking) |
| Graph ML      | PyTorch, PyTorch Geometric (GATv2Conv, LayerNorm)        |
| Matching      | fuzzywuzzy / python-Levenshtein                          |
| Packaging     | Nuitka (standalone Windows `.exe`)                       |

## Getting Started

> The trained models and OCR language model are **not** included in this
> repository. Re-train or restore them before the system can produce
> predictions (see the [training pipeline](#training-pipeline)).

```bash
# 1. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Required) Place the trained models at:
#    models/YOLO/best.pt
#    models/GNN/best.pt
#    and make the custom 'pof_ocr' Tesseract model available

# 4. Run the API server (host 0.0.0.0, port 5500)
python app.py
```

_The server will refuse to start if the model files are missing._

## API Reference

### `POST /pof`

Predicts the point of failure for a topology image.

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

**Error responses** (`status` set to `error` with a human-readable `message`):

- Invalid or empty base64 image
- Invalid image (undecodable, or no nodes detected)
- No `site_id` provided
- `site_id` not found in the image (no fuzzy match ≥ 70%)

### `GET /health`

```json
{ "message": "OK" }
```

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
