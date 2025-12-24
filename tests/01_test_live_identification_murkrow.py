import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest


def _jpeg_roundtrip(img: np.ndarray, quality: int = 70) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    assert ok
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert out is not None
    return out


def _downscale_like_web(img: np.ndarray, target_width: int = 500) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= 0:
        return img
    scale = target_width / w
    new_size = (target_width, max(1, int(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


@pytest.mark.slow
def test_live_like_murkrow_identifies_via_ocr_when_orb_unreliable():
    """
    Regression: in live camera mode, ORB can fail on noisy/perspective crops.
    OCR must still initialize (CPU OK) and identify Murkrow from the name strip.
    """
    pytest.importorskip("easyocr")

    db_path = Path("data/me1_me2_features.pkl")
    if not db_path.exists():
        pytest.skip("me1_me2_features.pkl not present")

    murkrow_path = Path("data/card_database/me2/Murkrow_me2-57.jpg")
    if not murkrow_path.exists():
        pytest.skip("Murkrow reference image missing")

    from src.identification.feature_matcher import FeatureMatcher

    matcher = FeatureMatcher(use_ocr=True, use_vector_db=False)
    matcher.load_database(str(db_path))

    # Make ORB path effectively inert for this test (fast) so we're validating OCR.
    matcher.card_features = {}

    img = cv2.imread(str(murkrow_path))
    assert img is not None

    # Simulate live pipeline degradation (downscale + jpeg + slight blur)
    live = _downscale_like_web(img, target_width=500)
    live = _jpeg_roundtrip(live, quality=70)
    live = cv2.GaussianBlur(live, (3, 3), 0)

    matches = matcher.identify(live, top_k=3)
    assert matches, "Expected OCR to return at least one match"

    best = matches[0]
    assert best["card_id"] == "me2-57"
    assert best["name"] == "Murkrow"







