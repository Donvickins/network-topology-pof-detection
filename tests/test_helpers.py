"""Unit tests for core/utils/helpers.py.

The OCR tests replace pytesseract.image_to_string with a fake function. This
keeps the tests fast and deterministic and means Tesseract does not need to be
installed to run them.
"""

import numpy as np
import pytest
import pytesseract

from core.utils import helpers
from core.utils.types import IMAGE, OCR, TextResult
from core.utils.exception_handler import InvalidImageException

# A tiny image is enough because the fake OCR ignores its contents.
IMG = np.zeros((1, 1), dtype=np.uint8)


@pytest.fixture(autouse=True)
def isolated_ocr(monkeypatch):
    """Keep every test away from the real Tesseract binary and config.yaml."""
    monkeypatch.setattr(helpers, "_configure_tesseract", lambda: None)
    monkeypatch.setattr(helpers, "TESS_LANGS", ["eng"])


# --- match_down_id -----------------------------------------------------------


def test_match_down_id_returns_exact_match():
    result = helpers.match_down_id("ABC123", ["ABC123"])

    assert result == [helpers.DownIdMatchResult(score=100.0, id="ABC123")]


def test_match_down_id_includes_fuzzy_match_above_threshold(monkeypatch):
    # Pin the threshold so this test does not depend on config.yaml.
    monkeypatch.setattr(helpers, "FUZZY_PERCENTAGE", 80)

    result = helpers.match_down_id("ABC123", ["ABC124"])

    assert [match.id for match in result] == ["ABC124"]
    assert result[0].score == pytest.approx(83.33, abs=0.01)


def test_match_down_id_excludes_match_below_threshold(monkeypatch):
    monkeypatch.setattr(helpers, "FUZZY_PERCENTAGE", 90)
    assert helpers.match_down_id("ABC123", ["ABC124"]) == []


def test_match_down_id_returns_empty_for_empty_down_id():
    assert helpers.match_down_id("", ["ABC123"]) == []


def test_match_down_id_returns_empty_for_no_node_ids():
    assert helpers.match_down_id("ABC123", []) == []


def test_match_down_id_sorts_matches_best_first(monkeypatch):
    monkeypatch.setattr(helpers, "FUZZY_PERCENTAGE", 0)

    result = helpers.match_down_id("ABC123", ["ABC124", "ABC123"])

    assert [match.id for match in result] == ["ABC123", "ABC124"]
    assert result[0].score > result[1].score


# --- extract_text ------------------------------------------------------------


def test_extract_text_returns_cleaned_id(monkeypatch):
    monkeypatch.setattr(
        helpers.pytesseract,
        "image_to_string",
        lambda img, config=None, lang=None: "ABC123\n",
    )

    result = helpers.extract_text(IMG)

    assert result == TextResult(text="ABC123", code=None, error=None)


@pytest.mark.parametrize(
    "raw_text, expected",
    [
        ("ABC123_extra", "ABC123"),
        ("ABC123-extra", "ABC123"),
    ],
)
def test_extract_text_keeps_only_prefix_before_separator(
    monkeypatch, raw_text, expected
):
    monkeypatch.setattr(
        helpers.pytesseract,
        "image_to_string",
        lambda img, config=None, lang=None: raw_text,
    )

    result = helpers.extract_text(IMG)

    assert result.text == expected
    assert result.code is None


def test_extract_text_falls_back_to_next_language(monkeypatch):
    monkeypatch.setattr(helpers, "TESS_LANGS", ["pof_ocr", "eng"])
    calls = []

    def fake_ocr(img, config=None, lang=None):
        calls.append(lang)
        if lang == "pof_ocr":
            raise pytesseract.TesseractError(1, "Failed loading language")
        return "SITE1"

    monkeypatch.setattr(helpers.pytesseract, "image_to_string", fake_ocr)

    result = helpers.extract_text(IMG)

    assert result.text == "SITE1"
    assert calls == ["pof_ocr", "eng"]


def test_extract_text_reports_missing_tesseract(monkeypatch):
    def fake_ocr(img, config=None, lang=None):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(helpers.pytesseract, "image_to_string", fake_ocr)

    result = helpers.extract_text(IMG)

    assert result.text is None
    assert result.code is OCR.TESSERACT_NOT_INSTALLED
    assert result.error == OCR.TESSERACT_NOT_INSTALLED.value

@pytest.mark.parametrize(
    "raw_text",["   \n",None,"","\n\n\n"],
)
def test_extract_text_reports_empty_result(monkeypatch, raw_text):
    monkeypatch.setattr(
        helpers.pytesseract,
        "image_to_string",
        lambda img, config=None, lang=None: raw_text,
    )

    result = helpers.extract_text(IMG)

    assert result.text is None
    assert result.code is OCR.OCR_EMPTY
    assert result.error == OCR.OCR_EMPTY.value


def test_extract_text_reports_tesseract_error_when_all_languages_fail(monkeypatch):
    monkeypatch.setattr(helpers, "TESS_LANGS", ["pof_ocr", "eng"])

    def fake_ocr(img, config=None, lang=None):
        raise pytesseract.TesseractError(1, "boom")

    monkeypatch.setattr(helpers.pytesseract, "image_to_string", fake_ocr)

    result = helpers.extract_text(IMG)

    assert result.text is None
    assert result.code is OCR.TESSERACT_ERROR
    assert result.error == OCR.TESSERACT_ERROR.value


# # --- site_id_img_2_binary ----------------------------------------------------


def test_site_id_img_2_binary_keeps_red_pixels(monkeypatch):
    # Disable upscaling so the expected shape stays easy to read.
    monkeypatch.setattr(helpers, "OCR_UPSCALE", 1.0)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)  # OpenCV uses BGR, so this is red.

    result = helpers.site_id_img_2_binary(img)

    assert result.shape == (10, 10)
    assert np.all(result == 255)


def test_site_id_img_2_binary_rejects_white_pixels(monkeypatch):
    monkeypatch.setattr(helpers, "OCR_UPSCALE", 1.0)
    img = np.full((10, 10, 3), 255, dtype=np.uint8)

    result = helpers.site_id_img_2_binary(img)

    assert result.shape == (10, 10)
    assert np.all(result == 0)


def test_site_id_img_2_binary_raises_for_none():
    with pytest.raises(InvalidImageException):
        helpers.site_id_img_2_binary(None)


# # --- crop_site_id_from_image -------------------------------------------------


def test_crop_site_id_from_image_crops_expected_region():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # bbox [50, 50, 100, 100] maps to rows 101:123 and columns 20:99.
    img[101:123, 20:99] = 200

    result = helpers.crop_site_id_from_image(img, [50, 50, 100, 100])

    assert result.code is None
    assert result.error is None
    assert result.site_id_image.shape == (22, 79, 3)
    assert np.all(result.site_id_image == 200)


def test_crop_site_id_from_image_rejects_inverted_bbox():
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    result = helpers.crop_site_id_from_image(img, [50, 50, 40, 100])

    assert result.site_id_image is None
    assert result.code is IMAGE.OUT_OF_BOUNDS
    assert result.error == IMAGE.OUT_OF_BOUNDS.value


def test_crop_site_id_from_image_rejects_crop_outside_image():
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # x_min=10 makes col_start negative (10 - LABEL_DX_LEFT).
    result = helpers.crop_site_id_from_image(img, [10, 50, 100, 100])

    assert result.site_id_image is None
    assert result.code is IMAGE.OUT_OF_BOUNDS


def test_crop_site_id_from_image_rejects_none():
    result = helpers.crop_site_id_from_image(None, [0, 0, 10, 10])

    assert result.site_id_image is None
    assert result.code is IMAGE.EMPTY_INPUT_IMAGE
    assert result.error == IMAGE.EMPTY_INPUT_IMAGE.value


def test_crop_site_id_from_image_rejects_missing_file(tmp_path):
    result = helpers.crop_site_id_from_image(tmp_path / "missing.png", [0, 0, 10, 10])

    assert result.site_id_image is None
    assert result.code is IMAGE.INVALID_FILE_PATH


# # --- create_node_tensor ------------------------------------------------------


def test_create_node_tensor_builds_features_and_down_flags(monkeypatch):
    monkeypatch.setattr(helpers, "FUZZY_PERCENTAGE", 80)
    nodes = [
        {"id": "ABC123", "type": "ATN", "color": "Red", "center": [10.0, 20.0]},
        {"id": "XYZ999", "type": "RTN", "color": "Green", "center": [30.0, 40.0]},
    ]

    result = helpers.create_node_tensor(nodes, down_id="ABC124")

    # 5 type features + 6 color features + 1 down flag = 12.
    assert result.x.shape == (2, 12)
    assert result.x[:, -1].tolist() == [1.0, 0.0]
    assert result.node_ids == ["ABC123", "XYZ999"]
    assert result.node_centers.shape == (2, 2)
    assert result.node_centers.tolist() == [[10.0, 20.0], [30.0, 40.0]]


def test_create_node_tensor_rejects_empty_node_list():
    with pytest.raises(InvalidImageException):
        helpers.create_node_tensor([], down_id="ABC123")


def test_create_node_tensor_marks_exact_down_id(monkeypatch):
    # With a 100% threshold only the exact match may be marked down.
    monkeypatch.setattr(helpers, "FUZZY_PERCENTAGE", 100)
    nodes = [
        {"id": "ABC123", "type": "ATN", "color": "Red", "center": [0.0, 0.0]},
        {"id": "ABC124", "type": "ATN", "color": "Red", "center": [10.0, 0.0]},
    ]

    result = helpers.create_node_tensor(nodes, down_id="ABC123")

    assert result.x[:, -1].tolist() == [1.0, 0.0]


# # --- create_edges_tensor -----------------------------------------------------


def test_create_edges_tensor_builds_both_directions(monkeypatch):
    monkeypatch.setattr(helpers, "MAX_DIST_THRESH", 50)
    node_centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    edges = [{"color": "Red", "endpoints": [(0.0, 0.0), (10.0, 0.0)]}]

    result = helpers.create_edges_tensor(edges, node_centers)

    assert result.edge_index.tolist() == [[0, 1], [1, 0]]
    assert result.edge_attr.shape == (2, 6)
    assert result.edge_attr.tolist() == [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2


def test_create_edges_tensor_discards_edge_with_far_endpoint(monkeypatch):
    monkeypatch.setattr(helpers, "MAX_DIST_THRESH", 50)
    node_centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    edges = [{"color": "Red", "endpoints": [(0.0, 0.0), (1000.0, 0.0)]}]

    result = helpers.create_edges_tensor(edges, node_centers)

    assert result.edge_index.shape == (2, 0)
    assert result.edge_attr.shape == (0, 6)


def test_create_edges_tensor_discards_edge_that_connects_same_node(monkeypatch):
    monkeypatch.setattr(helpers, "MAX_DIST_THRESH", 50)
    node_centers = np.array([[0.0, 0.0], [100.0, 0.0]])
    # Both endpoints are nearest to node 0.
    edges = [{"color": "Red", "endpoints": [(0.0, 0.0), (1.0, 0.0)]}]

    result = helpers.create_edges_tensor(edges, node_centers)

    assert result.edge_index.shape == (2, 0)
    assert result.edge_attr.shape == (0, 6)


def test_create_edges_tensor_returns_empty_for_no_node_centers():
    edges = [{"color": "Red", "endpoints": [(0.0, 0.0), (1.0, 0.0)]}]

    result = helpers.create_edges_tensor(edges, [])

    assert result.edge_index.shape == (2, 0)
    assert result.edge_attr.shape == (0, 6)


def test_create_edges_tensor_returns_empty_for_no_edges():
    result = helpers.create_edges_tensor([], np.array([[0.0, 0.0]]))

    assert result.edge_index.shape == (2, 0)
    assert result.edge_attr.shape == (0, 6)


# # --- is_demo_mode ------------------------------------------------------------


def test_is_demo_mode_true_when_env_is_demo(monkeypatch):
    monkeypatch.setenv("POF_MODEL_SOURCE", "demo")

    assert helpers.is_demo_mode() is True


def test_is_demo_mode_false_when_env_is_live(monkeypatch):
    monkeypatch.setenv("POF_MODEL_SOURCE", "live")

    assert helpers.is_demo_mode() is False


def test_is_demo_mode_false_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("POF_MODEL_SOURCE", raising=False)

    assert helpers.is_demo_mode() is False
