"""
부산시 문화축제 및 대규모 전시·컨벤션 데이터 정제(interim) 및 역사별 시공간 매칭(processed) 모듈

2단계 데이터 파이프라인:
1. build_festival_interim:
   원시 데이터(data/raw/festival/) 및 공식 연혁 보완 참조 테이블(data/reference/major_events_supplement.csv)에서
   불필요 데이터(타 지역, 소규모 개인 화랑, 상설 전시 등)를 제거하고 표준 스키마 및 예상 인원수 기반 규모 가중치를 산출하여
   data/interim/festival/festival_interim.parquet에 저장합니다.
2. build_festival_processed:
   정제된 interim 이벤트 데이터를 일별(date)로 전개하고 114개 도시철도 역사(station_master.parquet)와
   Haversine 거리 가중치(float), 크기형 최근접 역 순위(int: 3=1순위, 2=2순위, 1=3순위, 0=없음),
   규모×거리 복합 충격량 지수(float)를 계산하여 data/processed/festival_processed.parquet에 저장합니다.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from busan_metro_congestion.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REFERENCE_DATA_DIR,
)

# 대규모 다중이용 전시·문화시설 고정 좌표 및 인접 도시철도역 매핑
MAJOR_VENUE_COORDS: dict[str, tuple[float, float, str]] = {
    "벡스코": (35.1691, 129.1362, "센텀시티역 시립미술관역"),
    "BEXCO": (35.1691, 129.1362, "센텀시티역 시립미술관역"),
    "영화의전당": (35.1711, 129.1271, "센텀시티역"),
    "시립미술관": (35.1667, 129.1370, "시립미술관역 센텀시티역"),
    "부산현대미술관": (35.1092, 128.9426, "하단역"),
    "부산문화회관": (35.1274, 129.0935, "대연역 못골역"),
    "부산시민회관": (35.1384, 129.0652, "범일역 문현역"),
    "해운대해수욕장": (35.1587, 129.1604, "해운대역"),
    "광안리해수욕장": (35.1532, 129.1186, "광안역 금련산역"),
    "광안리": (35.1532, 129.1186, "광안역 금련산역"),
    "다대포해수욕장": (35.0469, 128.9662, "다대포해수욕장역"),
    "삼락생태공원": (35.1688, 128.9741, "사상역"),
    "화명생태공원": (35.2268, 129.0061, "화명역 구포역"),
    "대저생태공원": (35.2093, 128.9865, "강서구청역"),
    "맥도생태공원": (35.1645, 128.9602, "사상역"),
    "부산시민공원": (35.1681, 129.0573, "부전역 서면역"),
    "용두산공원": (35.1006, 129.0327, "남포역 중앙역"),
    "광복로": (35.0989, 129.0312, "남포역 자갈치역"),
    "자갈치": (35.0968, 129.0306, "자갈치역 남포역"),
    "송도해수욕장": (35.0758, 129.0169, "자갈치역 남포역"),
    "온천천": (35.1972, 129.0801, "교대역 동래역 부산대역"),
    "동래읍성": (35.2120, 129.0905, "동래역 명륜역"),
    "금정산성": (35.2420, 129.0925, "부산대역 장전역"),
    "북항": (35.1145, 129.0460, "부산역 중앙역"),
}

DEFAULT_FESTIVAL_VISITORS = 15_000
DEFAULT_EXHIBITION_VISITORS = 8_000
MAX_EXHIBITION_DURATION_DAYS = 120


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 WGS84 위경도 좌표 간 Haversine 구면 거리(km)를 계산합니다."""
    r_earth = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    )
    return 2.0 * r_earth * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def normalize_event_key(title: str) -> str:
    """행사명 매칭을 위해 연도, 회차, 괄호, 공백 등을 제거한 핵심 키를 반환합니다."""
    s = re.sub(r"\(.*?\)|〈.*?〉|<.*?>|\[.*?\]", "", str(title))
    s = re.sub(r"제\s*\d+\s*회", "", s)
    s = re.sub(r"20\d{2}\s*년?", "", s)
    s = re.sub(r"[^가-힣a-zA-Z0-9]", "", s)
    return s.lower().strip()


def _parse_standard_date(val: Any) -> date | None:
    """'YYYY-MM-DD' 또는 'YYYYMMDD' 문자열을 date 객체로 파싱합니다."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.startswith("0000"):
        return None
    m_dash = re.match(r"^(20\d{2})-(\d{1,2})-(\d{1,2})$", s)
    if m_dash:
        try:
            return date(
                int(m_dash.group(1)), int(m_dash.group(2)), int(m_dash.group(3))
            )
        except ValueError:
            return None
    m_compact = re.match(r"^(20\d{2})(\d{2})(\d{2})$", s)
    if m_compact:
        try:
            return date(
                int(m_compact.group(1)),
                int(m_compact.group(2)),
                int(m_compact.group(3)),
            )
        except ValueError:
            return None
    return None


def parse_informal_date_range(
    text: str, default_year: int | None = None
) -> tuple[date, date] | None:
    """
    부산축제정보 API 및 문체부 엑셀 등의 비정형 날짜 텍스트에서 (start_date, end_date)를 추출합니다.
    예:
    - '2025. 8. 1. ~ 8. 3.'
    - '2025년 11월 29일 ~ 2026년 1월 18일'
    - '23.12.31.~24.1.1.' (2자리 연도)
    - '5.3.~5.5.', '3.30.~31.', '12.31.' (default_year 제공 시 연도 생략 포맷)
    """
    if not text:
        return None
    s = str(text).strip()

    # 1. 4자리 연-월-일 ~ 4자리 연-월-일
    m_full = re.search(
        r"(20\d{2})\s*[.년/-]\s*(\d{1,2})\s*[.월/-]\s*(\d{1,2}).*?~\s*(20\d{2})\s*[.년/-]\s*(\d{1,2})\s*[.월/-]\s*(\d{1,2})",
        s,
        re.DOTALL,
    )
    if m_full:
        try:
            d1 = date(int(m_full.group(1)), int(m_full.group(2)), int(m_full.group(3)))
            d2 = date(int(m_full.group(4)), int(m_full.group(5)), int(m_full.group(6)))
            if d1 <= d2:
                return d1, d2
        except ValueError:
            pass

    # 2. 4자리 연-월-일 ~ 월-일 (종료 연도 생략)
    m_short = re.search(
        r"(20\d{2})\s*[.년/-]\s*(\d{1,2})\s*[.월/-]\s*(\d{1,2}).*?~\s*(\d{1,2})\s*[.월/-]\s*(\d{1,2})",
        s,
        re.DOTALL,
    )
    if m_short:
        try:
            y1 = int(m_short.group(1))
            m1, day1 = int(m_short.group(2)), int(m_short.group(3))
            m2, day2 = int(m_short.group(4)), int(m_short.group(5))
            d1 = date(y1, m1, day1)
            y2 = y1 + 1 if m2 < m1 else y1
            # '2025. 10. 23. ~ 10. 06.'과 같이 동일 월 내 종료일 십의 자리 오타(06 -> 26) 보정
            if y1 == y2 and m1 == m2 and day2 < day1 and day1 >= 20 and day2 < 10:
                day2 += 20
            d2 = date(y2, m2, day2)
            if d1 <= d2:
                return d1, d2
        except ValueError:
            pass

    # 3. 2자리 연.월.일 ~ 2자리 연.월.일 (예: '23.12.31.~24.1.1.')
    m_two_yr = re.search(
        r"(?<!\d)(\d{2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*~\s*(\d{2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})",
        s,
        re.DOTALL,
    )
    if m_two_yr:
        try:
            d1 = date(
                2000 + int(m_two_yr.group(1)),
                int(m_two_yr.group(2)),
                int(m_two_yr.group(3)),
            )
            d2 = date(
                2000 + int(m_two_yr.group(4)),
                int(m_two_yr.group(5)),
                int(m_two_yr.group(6)),
            )
            if d1 <= d2:
                return d1, d2
        except ValueError:
            pass

    # 4. 단일 4자리 연-월-일
    m_single = re.search(
        r"(20\d{2})\s*[.년/-]\s*(\d{1,2})\s*[.월/-]\s*(\d{1,2})",
        s,
    )
    if m_single:
        try:
            d1 = date(
                int(m_single.group(1)), int(m_single.group(2)), int(m_single.group(3))
            )
            return d1, d1
        except ValueError:
            pass

    # 5. default_year가 주어진 경우 연도 생략형 포맷 ('5.3.~5.5.', '3.30.~31.', '12.31.')
    if default_year is not None:
        m_md_md = re.search(
            r"(?<!\d)(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*~\s*(\d{1,2})\s*\.\s*(\d{1,2})",
            s,
        )
        if m_md_md:
            try:
                m1, day1 = int(m_md_md.group(1)), int(m_md_md.group(2))
                m2, day2 = int(m_md_md.group(3)), int(m_md_md.group(4))
                d1 = date(default_year, m1, day1)
                y2 = default_year + 1 if m2 < m1 else default_year
                d2 = date(y2, m2, day2)
                if d1 <= d2:
                    return d1, d2
            except ValueError:
                pass

        m_md_d = re.search(
            r"(?<!\d)(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*~\s*(\d{1,2})\s*\.?(?!\d)",
            s,
        )
        if m_md_d:
            try:
                m1, day1, day2 = (
                    int(m_md_d.group(1)),
                    int(m_md_d.group(2)),
                    int(m_md_d.group(3)),
                )
                d1 = date(default_year, m1, day1)
                d2 = date(default_year, m1, day2)
                if d1 <= d2:
                    return d1, d2
            except ValueError:
                pass

        m_md_single = re.match(r"^\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*$", s)
        if m_md_single:
            try:
                d1 = date(
                    default_year, int(m_md_single.group(1)), int(m_md_single.group(2))
                )
                return d1, d1
            except ValueError:
                pass

    return None


def _read_csv_auto(path: Path) -> pl.DataFrame:
    """다양한 인코딩(utf-8, cp949, euc-kr 등)의 CSV를 안전하게 로드합니다."""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "utf-16"):
        try:
            text = raw.decode(enc)
            return pl.read_csv(
                text.encode("utf-8"), infer_schema_length=10000, ignore_errors=True
            )
        except Exception:
            continue
    raise ValueError(f"CSV 인코딩을 해독할 수 없습니다: {path}")


def _parse_xlsx_sheet_rows(
    xlsx_path: Path, sheet_idx: int = 2
) -> list[list[str | None]]:
    """외부 의존성 없이 표준 zipfile/xml로 xlsx 시트 행들을 추출합니다."""
    with zipfile.ZipFile(xlsx_path) as z:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in tree.findall(
                "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"
            ):
                texts = [
                    n.text
                    for n in si.iter(
                        "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                    )
                    if n.text
                ]
                shared_strings.append("".join(texts))

        sheet_file = f"xl/worksheets/sheet{sheet_idx}.xml"
        if sheet_file not in z.namelist():
            return []

        sheet_tree = ET.fromstring(z.read(sheet_file))
        rows = sheet_tree.findall(
            ".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"
        )
        result: list[list[str | None]] = []
        for r in rows:
            row_dict: dict[int, str | None] = {}
            max_col = 0
            for c in r.findall(
                "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"
            ):
                ref = c.attrib.get("r", "")
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = 0
                for ch in col_letters:
                    col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
                max_col = max(max_col, col_idx)
                t_attr = c.attrib.get("t")
                v = c.find(
                    "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v"
                )
                val = v.text if v is not None else None
                if t_attr == "s" and val is not None:
                    val = shared_strings[int(val)]
                row_dict[col_idx] = val
            result.append([row_dict.get(i) for i in range(1, max_col + 1)])
        return result


def _match_venue_coords(place_text: str) -> tuple[float, float, str] | None:
    """장소명 문자열에서 주요 대규모 시설·장소 위경도 및 인접 역 정보를 찾습니다."""
    if not place_text:
        return None
    p_upper = str(place_text).upper()
    for keyword, info in MAJOR_VENUE_COORDS.items():
        if keyword in p_upper:
            return info
    return None


def _extract_mcst_visitors_and_events(
    mcst_dir: Path,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """
    문체부 지역축제 개최계획 엑셀(2022~2026)에서:
    1) 축제별 공식 전년도 방문객 수 룩업 사전
    2) 정확한 연-월-일이 기입된 부산 지역축제 레코드
    를 추출합니다.
    """
    visitor_lookup: dict[str, int] = {}
    mcst_events: list[dict[str, Any]] = []

    if not mcst_dir.exists():
        return visitor_lookup, mcst_events

    # 1) 2022~2024 엑셀: 단일 개최기간 텍스트(col 6) + 전년도 방문객 수(col 14)
    text_date_configs = [
        (2022, "2022년 지역축제 개최 계획(취합).xlsx"),
        (2023, "2023년 지역축제 개최 계획(취합).xlsx"),
        (2024, "2024년 지역축제 개최계획(홈페이지 게시)0930.xlsx"),
    ]
    for file_year, fname in text_date_configs:
        fpath = mcst_dir / fname
        if not fpath.exists():
            continue
        rows = _parse_xlsx_sheet_rows(fpath, sheet_idx=2)
        for r in rows:
            if len(r) <= 14:
                continue
            if not any(c and "부산" in str(c) for c in r[:4]):
                continue
            fest_name = str(r[4] or "").strip()
            date_text = str(r[6] or "").strip()
            place_name = str(r[7] or "").strip()
            if not fest_name or "취소" in fest_name or "취소" in date_text:
                continue

            key = normalize_event_key(fest_name)
            vis_str = str(r[14] or "0").replace(",", "").strip()
            vis_val = 0
            try:
                vis_val = int(float(vis_str))
            except ValueError:
                vis_val = 0

            if key and vis_val >= 1000:
                visitor_lookup[key] = max(visitor_lookup.get(key, 0), vis_val)

            parsed_range = parse_informal_date_range(date_text, default_year=file_year)
            if not parsed_range:
                continue
            d_start, d_end = parsed_range
            venue_info = _match_venue_coords(place_name) or _match_venue_coords(
                fest_name
            )
            if venue_info and d_start <= d_end:
                mcst_events.append(
                    {
                        "event_name": fest_name,
                        "category": "festival",
                        "venue_name": place_name,
                        "start_date": d_start,
                        "end_date": d_end,
                        "lat": venue_info[0],
                        "lng": venue_info[1],
                        "total_visitors": vis_val if vis_val >= 1000 else 0,
                        "trfc_info": venue_info[2],
                        "source": fname,
                        "priority": 3,
                        "source_year": file_year,
                    }
                )

    # 2) 2025~2026 엑셀: 시작/종료 연·월·일 분리 컬럼(col 11~16) + 전년도 방문객 수(col 26)
    split_date_configs = [
        (2025, "2025년 지역축제 개최계획 현황(0321).xlsx"),
        (2026, "2026년 지역축제 개최 계획 현황(공개용).xlsx"),
    ]
    for file_year, fname in split_date_configs:
        fpath = mcst_dir / fname
        if not fpath.exists():
            continue
        rows = _parse_xlsx_sheet_rows(fpath, sheet_idx=2)
        for r in rows:
            if len(r) <= 26:
                continue
            if not any(c and "부산" in str(c) for c in r[:4]):
                continue
            fest_name = str(r[4] or "").strip()
            place_name = str(r[6] or "").strip()
            if not fest_name or "취소" in fest_name:
                continue

            key = normalize_event_key(fest_name)
            vis_str = str(r[26] or "0").replace(",", "").strip()
            vis_val = 0
            try:
                vis_val = int(float(vis_str))
            except ValueError:
                vis_val = 0

            if key and vis_val > 0:
                visitor_lookup[key] = max(visitor_lookup.get(key, 0), vis_val)

            try:
                d_start = date(
                    int(float(str(r[11]))),
                    int(float(str(r[12]))),
                    int(float(str(r[13]))),
                )
                d_end = date(
                    int(float(str(r[14]))),
                    int(float(str(r[15]))),
                    int(float(str(r[16]))),
                )
            except (ValueError, TypeError):
                continue

            venue_info = _match_venue_coords(place_name) or _match_venue_coords(
                fest_name
            )
            if venue_info and d_start <= d_end:
                mcst_events.append(
                    {
                        "event_name": fest_name,
                        "category": "festival",
                        "venue_name": place_name,
                        "start_date": d_start,
                        "end_date": d_end,
                        "lat": venue_info[0],
                        "lng": venue_info[1],
                        "total_visitors": vis_val,
                        "trfc_info": venue_info[2],
                        "source": fname,
                        "priority": 3,
                        "source_year": file_year,
                    }
                )

    return visitor_lookup, mcst_events


def build_festival_interim(
    raw_festival_dir: Path | None = None,
    supplement_csv_path: Path | None = None,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """
    1단계(raw -> interim):
    원본 축제·전시 데이터 및 공식 연혁 보완 테이블에서 불필요 데이터를 제거하고
    규모 가중치(scale_weight)를 포함한 표준 이벤트 테이블을 생성하여
    data/interim/festival/festival_interim.parquet에 저장합니다.
    """
    if raw_festival_dir is None:
        raw_festival_dir = RAW_DATA_DIR / "festival"
    if supplement_csv_path is None:
        supplement_csv_path = REFERENCE_DATA_DIR / "major_events_supplement.csv"
    if output_path is None:
        output_path = INTERIM_DATA_DIR / "festival" / "festival_interim.parquet"

    records: list[dict[str, Any]] = []

    # 1. 문체부 지역축제 엑셀에서 공식 방문객 수 룩업 및 정형 일정 추출
    visitor_lookup, mcst_events = _extract_mcst_visitors_and_events(
        raw_festival_dir / "문화체육관광부_지역축제 정보"
    )
    records.extend(mcst_events)

    # 2. 공식 연혁 보완 참조 테이블 (major_events_supplement.csv) — 최우선 순위(priority=1)
    if supplement_csv_path.exists():
        df_supp = _read_csv_auto(supplement_csv_path)
        for r in df_supp.iter_rows(named=True):
            d_start = _parse_standard_date(r.get("start_date"))
            d_end = _parse_standard_date(r.get("end_date"))
            if d_start and d_end and d_start <= d_end:
                ename = str(r.get("event_name", "")).strip()
                vis = int(r.get("total_visitors") or 0)
                key = normalize_event_key(ename)
                if key and vis > 0:
                    visitor_lookup[key] = max(visitor_lookup.get(key, 0), vis)
                records.append(
                    {
                        "event_name": ename,
                        "category": str(r.get("category", "festival")).strip(),
                        "venue_name": str(r.get("venue_name", "")).strip(),
                        "start_date": d_start,
                        "end_date": d_end,
                        "lat": float(r["lat"]),
                        "lng": float(r["lng"]),
                        "total_visitors": vis,
                        "trfc_info": str(r.get("trfc_info") or "").strip(),
                        "source": "major_events_supplement.csv",
                        "priority": 1,
                        "source_year": d_start.year,
                    }
                )

    # 3. 부산축제정보 OpenAPI (busan_festival_kr.json) 로드 -> 좌표/교통 룩업 및 일정 추출
    api_coord_lookup: dict[str, tuple[float, float, str]] = {}
    fest_json_path = raw_festival_dir / "busan_festival_kr.json"
    if fest_json_path.exists():
        payload = json.loads(fest_json_path.read_text(encoding="utf-8"))
        items = payload.get("getFestivalKr", {}).get("item", [])
        if isinstance(items, dict):
            items = [items]
        for it in items:
            raw_title = str(it.get("MAIN_TITLE", "")).strip()
            clean_title = re.sub(r"\(한.*?\)", "", raw_title).strip()
            if not clean_title:
                continue
            lat = float(it.get("LAT") or 0.0)
            lng = float(it.get("LNG") or 0.0)
            trfc = str(it.get("TRFC_INFO") or "").strip()
            key = normalize_event_key(clean_title)
            if lat > 34.0 and lng > 128.0 and key:
                api_coord_lookup[key] = (lat, lng, trfc)

            day_text = str(
                it.get("USAGE_DAY") or it.get("USAGE_DAY_WEEK_AND_TIME") or ""
            )
            parsed_range = parse_informal_date_range(day_text)
            if parsed_range and lat > 34.0 and lng > 128.0:
                d_start, d_end = parsed_range
                if (d_end - d_start).days <= MAX_EXHIBITION_DURATION_DAYS:
                    records.append(
                        {
                            "event_name": clean_title,
                            "category": "festival",
                            "venue_name": str(it.get("PLACE") or clean_title).strip(),
                            "start_date": d_start,
                            "end_date": d_end,
                            "lat": lat,
                            "lng": lng,
                            "total_visitors": visitor_lookup.get(key, 0),
                            "trfc_info": trfc,
                            "source": "busan_festival_kr.json",
                            "priority": 2,
                            "source_year": d_start.year,
                        }
                    )

    # 4. KC_488 전국 문화축제 표준데이터 (2019~2025, 7개 파일)
    kc_dir = (
        raw_festival_dir
        / "전국 문화축제 표준데이터(시군구 행정동 100미터격자코드 포함)"
    )
    if kc_dir.exists():
        for f in sorted(kc_dir.glob("*.csv")):
            m_year = re.search(r"(20\d{2})", f.stem)
            file_year = int(m_year.group(1)) if m_year else 0
            df_kc = _read_csv_auto(f)
            if "CTPRVN_NM" not in df_kc.columns:
                continue
            df_busan = df_kc.filter(pl.col("CTPRVN_NM") == "부산광역시")
            for r in df_busan.iter_rows(named=True):
                ename = str(r.get("FCLTY_NM") or "").strip()
                fest_cn = str(r.get("FSTVL_CN") or "").strip()
                if "취소" in ename or "잠정연기" in ename or "축제취소" in fest_cn:
                    continue
                d_start = _parse_standard_date(r.get("FSTVL_BEGIN_DE"))
                d_end = _parse_standard_date(r.get("FSTVL_END_DE"))
                lat = float(r.get("FCLTY_LA") or 0.0)
                lng = float(r.get("FCLTY_LO") or 0.0)
                # 2019~2021년 파일에서 FCLTY_LA(위도)와 FCLTY_LO(경도)가 반전된 경우 교정
                if lat > 100.0 and lng < 50.0:
                    lat, lng = lng, lat
                if not (ename and d_start and d_end and d_start <= d_end):
                    continue
                if not (34.8 <= lat <= 35.5 and 128.7 <= lng <= 129.4):
                    continue
                # 차년도 일정 미확정 상태에서 매월 1일 단발로 가기입된 플레이스홀더 행 제거
                if (
                    file_year
                    and d_start.year > file_year
                    and d_start == d_end
                    and d_start.day == 1
                ):
                    continue
                key = normalize_event_key(ename)
                trfc = api_coord_lookup.get(key, (lat, lng, ""))[2]
                records.append(
                    {
                        "event_name": ename,
                        "category": "festival",
                        "venue_name": str(r.get("OPMTN_PLACE_NM") or "").strip(),
                        "start_date": d_start,
                        "end_date": d_end,
                        "lat": lat,
                        "lng": lng,
                        "total_visitors": visitor_lookup.get(key, 0),
                        "trfc_info": trfc,
                        "source": f.name,
                        "priority": 2,
                        "source_year": file_year,
                    }
                )

    # 5. 전국문화축제표준데이터.csv
    std_csv = raw_festival_dir / "전국문화축제표준데이터.csv"
    if std_csv.exists():
        df_std = _read_csv_auto(std_csv)
        df_std_b = df_std.filter(
            pl.col("소재지도로명주소").fill_null("").str.contains("부산")
            | pl.col("소재지지번주소").fill_null("").str.contains("부산")
            | pl.col("제공기관명").fill_null("").str.contains("부산")
        )
        for r in df_std_b.iter_rows(named=True):
            ename = str(r.get("축제명") or "").strip()
            d_start = _parse_standard_date(r.get("축제시작일자"))
            d_end = _parse_standard_date(r.get("축제종료일자"))
            lat = float(r.get("위도") or 0.0)
            lng = float(r.get("경도") or 0.0)
            if not (ename and d_start and d_end and d_start <= d_end):
                continue
            if not (34.8 <= lat <= 35.5 and 128.7 <= lng <= 129.4):
                continue
            key = normalize_event_key(ename)
            trfc = api_coord_lookup.get(key, (lat, lng, ""))[2]
            records.append(
                {
                    "event_name": ename,
                    "category": "festival",
                    "venue_name": str(r.get("개최장소") or "").strip(),
                    "start_date": d_start,
                    "end_date": d_end,
                    "lat": lat,
                    "lng": lng,
                    "total_visitors": visitor_lookup.get(key, 0),
                    "trfc_info": trfc,
                    "source": "전국문화축제표준데이터.csv",
                    "priority": 2,
                    "source_year": 2025,
                }
            )

    # 6. 산업통상부_전시사업(국내 전시회)_20241231.csv (벡스코 개최분)
    motie_csv = raw_festival_dir / "산업통상부_전시사업(국내 전시회)_20241231.csv"
    if motie_csv.exists():
        df_motie = _read_csv_auto(motie_csv)
        for r in df_motie.iter_rows(named=True):
            venue = str(r.get("전시장소") or "").strip()
            if "벡스코" not in venue and "BEXCO" not in venue.upper():
                continue
            ename = str(r.get("전시회명") or "").strip()
            d_start = _parse_standard_date(r.get("전시시작일"))
            d_end = _parse_standard_date(r.get("전시종료일"))
            if not (ename and d_start and d_end and d_start <= d_end):
                continue
            vis = int(r.get("참관객") or 0)
            key = normalize_event_key(ename)
            if key and vis > 0:
                visitor_lookup[key] = max(visitor_lookup.get(key, 0), vis)
            bexco_lat, bexco_lng, bexco_trfc = MAJOR_VENUE_COORDS["벡스코"]
            records.append(
                {
                    "event_name": ename,
                    "category": "exhibition",
                    "venue_name": venue,
                    "start_date": d_start,
                    "end_date": d_end,
                    "lat": bexco_lat,
                    "lng": bexco_lng,
                    "total_visitors": vis,
                    "trfc_info": bexco_trfc,
                    "source": motie_csv.name,
                    "priority": 1,
                    "source_year": 2025,
                }
            )

    # 7. 부산광역시_전시 목록 서비스 (busan_culture_exhibit.json) — 대규모 시설 필터링
    exhibit_json = raw_festival_dir / "busan_culture_exhibit.json"
    if exhibit_json.exists():
        payload = json.loads(exhibit_json.read_text(encoding="utf-8"))
        items = payload.get("item") or payload.get("getBusanCultureExhibit", {}).get(
            "item", []
        )
        if isinstance(items, dict):
            items = [items]
        for it in items:
            place_nm = str(it.get("place_nm") or "").strip()
            venue_info = _match_venue_coords(place_nm)
            if venue_info is None:
                continue
            ename = str(it.get("title") or "").strip()
            d_start = _parse_standard_date(it.get("op_st_dt"))
            d_end = _parse_standard_date(it.get("op_ed_dt"))
            if not (ename and d_start and d_end and d_start <= d_end):
                continue
            duration = (d_end - d_start).days + 1
            if duration > MAX_EXHIBITION_DURATION_DAYS:
                continue
            key = normalize_event_key(ename)
            records.append(
                {
                    "event_name": ename,
                    "category": "exhibition",
                    "venue_name": place_nm,
                    "start_date": d_start,
                    "end_date": d_end,
                    "lat": venue_info[0],
                    "lng": venue_info[1],
                    "total_visitors": visitor_lookup.get(key, 0),
                    "trfc_info": venue_info[2],
                    "source": "busan_culture_exhibit.json",
                    "priority": 3,
                    "source_year": 2025,
                }
            )

    if not records:
        empty_df = pl.DataFrame(
            schema={
                "event_name": pl.Utf8,
                "category": pl.Utf8,
                "venue_name": pl.Utf8,
                "start_date": pl.Date,
                "end_date": pl.Date,
                "duration_days": pl.Int32,
                "lat": pl.Float64,
                "lng": pl.Float64,
                "total_visitors": pl.Int64,
                "daily_visitors": pl.Float64,
                "scale_weight": pl.Float64,
                "trfc_info": pl.Utf8,
                "source": pl.Utf8,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df.write_parquet(output_path)
        return empty_df

    # 8. 중복 제거 및 일평균 방문객 수 / 규모 가중치(scale_weight) 계산
    # 동일 연도 내 동일 행사명(정규화 키) 중복 시 우선순위·최신 파일·방문객 수 기준으로 단일화
    dedup_map: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in sorted(
        records,
        key=lambda x: (
            x["priority"],
            -int(x.get("source_year", 0)),
            -int(x["total_visitors"]),
            -((x["end_date"] - x["start_date"]).days),
        ),
    ):
        norm_key = normalize_event_key(rec["event_name"])
        if not norm_key:
            norm_key = rec["event_name"]
        dedup_key = (norm_key, rec["start_date"].year)
        if dedup_key not in dedup_map:
            dedup_map[dedup_key] = rec
        else:
            existing = dedup_map[dedup_key]
            if not existing["trfc_info"] and rec["trfc_info"]:
                existing["trfc_info"] = rec["trfc_info"]
            if existing["total_visitors"] == 0 and rec["total_visitors"] > 0:
                existing["total_visitors"] = rec["total_visitors"]

    final_rows: list[dict[str, Any]] = []
    for (norm_key, _), rec in dedup_map.items():
        duration_days = max(1, (rec["end_date"] - rec["start_date"]).days + 1)
        total_vis = int(rec["total_visitors"])
        if total_vis <= 0:
            total_vis = visitor_lookup.get(norm_key, 0)
        if total_vis <= 0:
            total_vis = (
                DEFAULT_FESTIVAL_VISITORS
                if rec["category"] == "festival"
                else DEFAULT_EXHIBITION_VISITORS
            )
        daily_vis = float(total_vis) / float(duration_days)
        scale_w = math.log10(1.0 + daily_vis)
        final_rows.append(
            {
                "event_name": rec["event_name"],
                "category": rec["category"],
                "venue_name": rec["venue_name"],
                "start_date": rec["start_date"],
                "end_date": rec["end_date"],
                "duration_days": duration_days,
                "lat": float(rec["lat"]),
                "lng": float(rec["lng"]),
                "total_visitors": total_vis,
                "daily_visitors": round(daily_vis, 2),
                "scale_weight": round(scale_w, 6),
                "trfc_info": rec["trfc_info"],
                "source": rec["source"],
            }
        )

    df_interim = (
        pl.DataFrame(final_rows)
        .with_columns(
            [
                pl.col("duration_days").cast(pl.Int32),
                pl.col("total_visitors").cast(pl.Int64),
                pl.col("daily_visitors").cast(pl.Float64),
                pl.col("scale_weight").cast(pl.Float64),
            ]
        )
        .sort(["start_date", "event_name"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_interim.write_parquet(output_path)
    return df_interim


def build_festival_processed(
    interim_df: pl.DataFrame | None = None,
    interim_path: Path | None = None,
    station_master_path: Path | None = None,
    output_path: Path | None = None,
    max_radius_km: float = 1.5,
) -> pl.DataFrame:
    """
    2단계(interim -> processed):
    정제된 축제·전시 이벤트 테이블과 114개 도시철도 역사 마스터(station_master.parquet)를
    시공간 매칭하여 역사별·일자별 파생 피처 테이블을 생성하고
    data/processed/festival_processed.parquet에 저장합니다.

    거리 순위(nearest_event_rank)는 크기형 int8로 부여합니다:
    - 3: 행사장 최인접 1순위 역 (또는 TRFC_INFO 직접 명시 역)
    - 2: 행사장 인접 2순위 역
    - 1: 행사장 인접 3순위 이하(반경 내) 역
    - 0: 영향권 밖 또는 행사 없음
    """
    if interim_df is None:
        if interim_path is None:
            interim_path = INTERIM_DATA_DIR / "festival" / "festival_interim.parquet"
        if not interim_path.exists():
            interim_df = build_festival_interim(output_path=interim_path)
        else:
            interim_df = pl.read_parquet(interim_path)

    if station_master_path is None:
        station_master_path = REFERENCE_DATA_DIR / "station_master.parquet"
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "festival_processed.parquet"

    if not station_master_path.exists():
        raise FileNotFoundError(
            f"역사 마스터 Parquet 파일을 찾을 수 없습니다: {station_master_path}"
        )

    df_stations = pl.read_parquet(station_master_path)
    stations = df_stations.select(
        ["station_id", "station_name", "lat", "lng"]
    ).to_dicts()

    daily_records: list[dict[str, Any]] = []

    for ev in interim_df.iter_rows(named=True):
        ev_lat = float(ev["lat"])
        ev_lng = float(ev["lng"])
        scale_w = float(ev["scale_weight"])
        trfc_text = str(ev.get("trfc_info") or "")
        d_start: date = ev["start_date"]
        d_end: date = ev["end_date"]

        # 1. 이 행사에 대해 반경 내 또는 TRFC_INFO 언급 역사 탐색
        matched_stations: list[tuple[int, float, bool]] = []
        for st in stations:
            sid = int(st["station_id"])
            sname = str(st["station_name"])
            dist_km = haversine_km(ev_lat, ev_lng, float(st["lat"]), float(st["lng"]))

            # TRFC_INFO에 '다대포해수욕장역' 또는 '다대포해수욕장' 언급 여부 확인
            search_token = sname if sname.endswith("역") else f"{sname}역"
            is_trfc_mentioned = bool(trfc_text and search_token in trfc_text)

            if dist_km <= max_radius_km or is_trfc_mentioned:
                matched_stations.append((sid, dist_km, is_trfc_mentioned))

        if not matched_stations:
            continue

        # 2. 거리 오름차순 정렬하여 순위 산정 (TRFC_INFO 언급 역은 최우선 정렬)
        matched_stations.sort(key=lambda x: (0 if x[2] else 1, x[1]))

        station_metrics: list[tuple[int, int, float, float, float]] = []
        for idx, (sid, dist_km, _is_trfc) in enumerate(matched_stations):
            if idx == 0:
                rank_score = 3  # 1순위 최인접역
            elif idx == 1:
                rank_score = 2  # 2순위 인접역
            else:
                rank_score = 1  # 3순위 이하 반경 내 인접역

            dist_w = 1.0 / (1.0 + dist_km)
            impact = scale_w * dist_w
            station_metrics.append((sid, rank_score, dist_km, dist_w, impact))

        # 3. 시작일 ~ 종료일 일별 전개
        cur_date = d_start
        while cur_date <= d_end:
            for sid, rank_score, dist_km, dist_w, impact in station_metrics:
                daily_records.append(
                    {
                        "station_id": sid,
                        "date": cur_date,
                        "rank_score": rank_score,
                        "dist_km": dist_km,
                        "dist_weight": dist_w,
                        "scale_weight": scale_w,
                        "impact_score": impact,
                    }
                )
            cur_date += timedelta(days=1)

    if not daily_records:
        empty_proc = pl.DataFrame(
            schema={
                "station_id": pl.Int64,
                "date": pl.Date,
                "is_festival": pl.Int8,
                "event_count": pl.Int8,
                "nearest_event_rank": pl.Int8,
                "min_event_dist_km": pl.Float64,
                "max_dist_weight": pl.Float64,
                "max_scale_weight": pl.Float64,
                "event_impact_score": pl.Float64,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty_proc.write_parquet(output_path)
        return empty_proc

    df_daily = pl.DataFrame(daily_records)
    df_processed = (
        df_daily.group_by(["station_id", "date"])
        .agg(
            [
                pl.lit(1).cast(pl.Int8).alias("is_festival"),
                pl.len().cast(pl.Int8).alias("event_count"),
                pl.col("rank_score").max().cast(pl.Int8).alias("nearest_event_rank"),
                pl.col("dist_km").min().round(4).alias("min_event_dist_km"),
                pl.col("dist_weight").max().round(6).alias("max_dist_weight"),
                pl.col("scale_weight").max().round(6).alias("max_scale_weight"),
                pl.col("impact_score").sum().round(6).alias("event_impact_score"),
            ]
        )
        .sort(["station_id", "date"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.write_parquet(output_path)
    return df_processed


def process_festival_data(
    file_path: Path | str | None = None,
    station_master_path: Path | None = None,
    interim_output_path: Path | None = None,
    processed_output_path: Path | None = None,
    max_radius_km: float = 1.5,
) -> pl.DataFrame:
    """
    축제 및 대규모 전시 데이터 2단계 전처리 통합 진입점:
    1) raw -> data/interim/festival/festival_interim.parquet 생성
    2) interim -> data/processed/festival_processed.parquet 생성 및 반환
    """
    raw_dir = Path(file_path) if file_path is not None else None
    df_interim = build_festival_interim(
        raw_festival_dir=raw_dir,
        output_path=interim_output_path,
    )
    return build_festival_processed(
        interim_df=df_interim,
        station_master_path=station_master_path,
        output_path=processed_output_path,
        max_radius_km=max_radius_km,
    )


if __name__ == "__main__":
    df_out = process_festival_data()
    print(f"[성공] 축제·전시 processed 생성 완료: {len(df_out):,}행")
