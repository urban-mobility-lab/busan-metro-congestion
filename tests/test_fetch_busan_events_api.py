import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.fetch_busan_events_api import (
    fetch_busan_culture_exhibit,
    fetch_busan_festival_kr,
    fetch_paginated_items,
    get_api_key,
)


def test_get_api_key_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PUBLIC_DATA_PORTAL_KEY", raising=False)
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="PUBLIC_DATA_PORTAL_KEY"):
        get_api_key(env_path=empty_env)


def test_get_api_key_from_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PUBLIC_DATA_PORTAL_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("PUBLIC_DATA_PORTAL_KEY=test_dummy_key_123\n", encoding="utf-8")

    assert get_api_key(env_path=env_file) == "test_dummy_key_123"


def test_fetch_paginated_items_multi_page() -> None:
    page1_payload = {
        "getBusanCultureExhibit": {
            "header": {"code": "00", "message": "NORMAL_CODE"},
            "totalCount": 3,
            "item": [
                {"res_no": "1", "title": "전시A"},
                {"res_no": "2", "title": "전시B"},
            ],
        }
    }
    page2_payload = {
        "getBusanCultureExhibit": {
            "header": {"code": "00", "message": "NORMAL_CODE"},
            "totalCount": 3,
            "item": [
                {"res_no": "3", "title": "전시C"},
            ],
        }
    }

    mock_resp1 = MagicMock()
    mock_resp1.json.return_value = page1_payload
    mock_resp2 = MagicMock()
    mock_resp2.json.return_value = page2_payload

    with patch(
        "scripts.fetch_busan_events_api.requests.get",
        side_effect=[mock_resp1, mock_resp2],
    ) as mock_get:
        result = fetch_paginated_items(
            url="https://example.com/api",
            root_key="getBusanCultureExhibit",
            api_key="dummy_key",
            num_of_rows=2,
        )

    assert mock_get.call_count == 2
    body = result["getBusanCultureExhibit"]
    assert body["totalCount"] == 3
    assert len(body["item"]) == 3
    assert [it["res_no"] for it in body["item"]] == ["1", "2", "3"]


def test_fetch_paginated_items_api_error() -> None:
    err_payload = {
        "getFestivalKr": {
            "header": {"code": "30", "message": "SERVICE_KEY_IS_NOT_REGISTERED_ERROR"},
            "totalCount": 0,
            "item": [],
        }
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = err_payload

    with patch("scripts.fetch_busan_events_api.requests.get", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="OpenAPI 응답 오류"):
            fetch_paginated_items(
                url="https://example.com/api",
                root_key="getFestivalKr",
                api_key="invalid_key",
                num_of_rows=10,
            )


def test_fetch_festival_and_exhibit_save_json(tmp_path: Path) -> None:
    fest_payload = {
        "getFestivalKr": {
            "header": {"code": "00", "message": "NORMAL_CODE"},
            "totalCount": 1,
            "item": [{"UC_SEQ": 101, "MAIN_TITLE": "부산불꽃축제"}],
        }
    }
    ex_payload = {
        "getBusanCultureExhibit": {
            "header": {"code": "00", "message": "NORMAL_CODE"},
            "totalCount": 1,
            "item": [
                {
                    "res_no": "202211",
                    "title": "국제게임전시회 지스타 2022",
                    "place_nm": "벡스코",
                }
            ],
        }
    }

    mock_fest_resp = MagicMock()
    mock_fest_resp.json.return_value = fest_payload
    mock_ex_resp = MagicMock()
    mock_ex_resp.json.return_value = ex_payload

    fest_out = tmp_path / "busan_festival_kr.json"
    ex_out = tmp_path / "busan_culture_exhibit.json"

    with patch(
        "scripts.fetch_busan_events_api.requests.get",
        side_effect=[mock_fest_resp, mock_ex_resp],
    ):
        fetch_busan_festival_kr(api_key="dummy", output_path=fest_out)
        fetch_busan_culture_exhibit(api_key="dummy", output_path=ex_out)

    assert fest_out.exists()
    assert ex_out.exists()

    fest_saved = json.loads(fest_out.read_text(encoding="utf-8"))
    assert fest_saved["getFestivalKr"]["totalCount"] == 1
    assert fest_saved["getFestivalKr"]["item"][0]["MAIN_TITLE"] == "부산불꽃축제"

    ex_saved = json.loads(ex_out.read_text(encoding="utf-8"))
    assert ex_saved["getBusanCultureExhibit"]["totalCount"] == 1
    assert ex_saved["getBusanCultureExhibit"]["item"][0]["place_nm"] == "벡스코"
