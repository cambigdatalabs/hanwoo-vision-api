from __future__ import annotations

import base64

from hanwoo.core.image_payload import encode_image_payload
from hanwoo.core.image_payload import attach_image_payload


def test_encode_image_payload(tmp_path) -> None:
    image_path = tmp_path / "match.png"
    image_path.write_bytes(b"image-bytes")

    payload = encode_image_payload(image_path)

    assert payload == {
        "image_mime_type": "image/png",
        "image_size_bytes": 11,
        "image_base64": base64.b64encode(b"image-bytes").decode("ascii"),
    }


def test_attach_match_image_adds_payload_to_one_match(tmp_path) -> None:
    image_path = tmp_path / "match.jpg"
    image_path.write_bytes(b"top-match")
    match = {"rank": 1, "image_path": str(image_path)}

    result = attach_image_payload(match)

    assert result["rank"] == 1
    assert result["image_path"] == str(image_path)
    assert result["image_mime_type"] == "image/jpeg"
    assert result["image_size_bytes"] == 9
    assert result["image_base64"] == base64.b64encode(b"top-match").decode("ascii")
