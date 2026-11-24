"""
부산시 문화축제 및 대규모 전시·컨벤션 2단계 전처리 모듈(festival.py) 단위 테스트
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from busan_metro_congestion.config import REFERENCE_DATA_DIR, SAMPLES_DIR
from busan_metro_congestion.preprocess.festival import (
    build_festival_interim,
    build_festival_processed,
    haversine_km,
    normalize_event_key,
    parse_informal_date_range,
    process_festival_data,
)


def test_haversine_km_accuracy() -> None:
    """동일 좌표 거리는 0이며, 센텀시티역~벡스코 거리는 약 0.2~0.3km 범위여야 합니다."""
    assert haversine_km(35.1691, 129.1362, 35.1691, 129.1362) == pytest.approx(
        0.0, abs=1e-6
    )
    # 센텀시티역(35.1688, 129.1317) ~ 벡스코(35.1691, 129.1362)
    dist = haversine_km(35.1688, 129.1317, 35.1691, 129.1362)
    assert 0.2 <= dist <= 0.5


def test_normalize_event_key() -> None:
    """연도, 회차, 괄호, 특수문자를 제거하고 핵심 행사명만 소문자로 정규화해야 합니다."""
    assert normalize_event_key(
        "2024년 제19회 부산불꽃축제(한,영)"
    ) == normalize_event_key("부산불꽃축제")
    assert normalize_event_key("제29회 부산국제영화제 [BIFF]") == "부산국제영화제"
    assert normalize_event_key("G-STAR 2025 (지스타)") == "gstar"


def test_parse_informal_date_range() -> None:
    """비정형 날짜 문자열에서 시작일과 종료일을 정확히 추출해야 합니다."""
    assert parse_informal_date_range("2025. 8. 1. ~ 8. 3.") == (
        date(2025, 8, 1),
        date(2025, 8, 3),
    )
    assert parse_informal_date_range("2025년 11월 29일 ~ 2026년 1월 18일") == (
        date(2025, 11, 29),
        date(2026, 1, 18),
    )
    assert parse_informal_date_range("2025. 11. 15.(토)") == (
        date(2025, 11, 15),
        date(2025, 11, 15),
    )
    assert parse_informal_date_range("23.12.31.\n~24.1.1.") == (
        date(2023, 12, 31),
        date(2024, 1, 1),
    )
    assert parse_informal_date_range("5.3.~5.5.", default_year=2024) == (
        date(2024, 5, 3),
        date(2024, 5, 5),
    )
    assert parse_informal_date_range("3.30.~31.", default_year=2024) == (
        date(2024, 3, 30),
        date(2024, 3, 31),
    )
    assert parse_informal_date_range("2025. 10. 23. ~ 10. 06.") == (
        date(2025, 10, 23),
        date(2025, 10, 26),
    )
    assert parse_informal_date_range("상시 운영") is None


def test_major_events_supplement_reference_exists() -> None:
    """data/reference/major_events_supplement.csv 참조 파일의 존재 및 필수 대표 행사 포함 여부를 검증합니다."""
    supp_path = REFERENCE_DATA_DIR / "major_events_supplement.csv"
    assert supp_path.exists(), "major_events_supplement.csv 파일이 존재해야 합니다."

    df = pl.read_csv(supp_path)
    assert len(df) >= 50
    required_cols = {
        "event_name",
        "category",
        "venue_name",
        "start_date",
        "end_date",
        "lat",
        "lng",
        "total_visitors",
        "trfc_info",
    }
    assert required_cols.issubset(set(df.columns))

    names_joined = " ".join(df["event_name"].to_list())
    for keyword in (
        "지스타",
        "모터쇼",
        "모빌리티쇼",
        "부산국제영화제",
        "부산불꽃축제",
        "부산국제록페스티벌",
    ):
        assert keyword in names_joined, f"대표 행사 '{keyword}'가 누락되었습니다."


def test_build_festival_interim_filtering_and_scale_weight(tmp_path: Path) -> None:
    """
    1단계(build_festival_interim)가 불필요 행(타 지역, 차년도 1일 플레이스홀더, 소규모 화랑, 상설 전시)을
    제거하고 위경도 반전(2019~2021) 교정 및 규모 가중치(scale_weight)를 정확히 산출하는지 검증합니다.
    """
    raw_dir = tmp_path / "raw_festival"
    kc_dir = raw_dir / "전국 문화축제 표준데이터(시군구 행정동 100미터격자코드 포함)"
    kc_dir.mkdir(parents=True, exist_ok=True)

    # 1. KC_488_2023 샘플: 정상 부산 축제 1건 + 차년도 1일 플레이스홀더 오기입 1건 + 위경도 반전 1건 + 타 시도 1건
    pl.DataFrame(
        {
            "FCLTY_NM": [
                "제18회 해운대모래축제",
                "제11회 수영강변축제",
                "2019 금정산성축제",
                "서울빛초롱축제",
            ],
            "CTPRVN_NM": ["부산광역시", "부산광역시", "부산광역시", "서울특별시"],
            "OPMTN_PLACE_NM": ["해운대해수욕장", "해운대구청", "금정산성", "청계천"],
            "FSTVL_BEGIN_DE": ["2023-05-19", "2024-10-01", "20190524", "2023-12-15"],
            "FSTVL_END_DE": ["2023-05-22", "2024-10-01", "20190526", "2023-12-31"],
            "FCLTY_LA": [35.1587, 35.1631, 129.0635, 37.5665],
            "FCLTY_LO": [129.1604, 129.1635, 35.2453, 126.9780],
        }
    ).write_csv(kc_dir / "KC_488_WNTY_CLTFSTVL_2023.csv")

    # 2. 전시 목록 API JSON 샘플: 벡스코 단기 전시(유효) + 개인 갤러리(제외) + 120일 초과 상설전시(제외)
    exhibit_payload = {
        "item": [
            {
                "title": "부산국제보트쇼",
                "place_nm": "벡스코",
                "op_st_dt": "2024-04-19",
                "op_ed_dt": "2024-04-21",
            },
            {
                "title": "소규모 개인 초대전",
                "place_nm": "소울아트스페이스",
                "op_st_dt": "2024-04-01",
                "op_ed_dt": "2024-04-15",
            },
            {
                "title": "박물관 연간 상설전시",
                "place_nm": "시립미술관",
                "op_st_dt": "2024-01-01",
                "op_ed_dt": "2024-12-31",
            },
        ]
    }
    (raw_dir / "busan_culture_exhibit.json").write_text(
        json.dumps(exhibit_payload, ensure_ascii=False), encoding="utf-8"
    )

    # 3. 보완 테이블 샘플: 지스타 2024 (4일간 215,000명)
    supp_csv = tmp_path / "supplement.csv"
    pl.DataFrame(
        {
            "event_name": ["지스타 2024 (G-STAR)"],
            "category": ["exhibition"],
            "venue_name": ["벡스코"],
            "start_date": ["2024-11-14"],
            "end_date": ["2024-11-17"],
            "lat": [35.1691],
            "lng": [129.1362],
            "total_visitors": [215000],
            "trfc_info": ["센텀시티역 시립미술관역"],
            "source_note": ["한국전시산업진흥회"],
        }
    ).write_csv(supp_csv)

    interim_out = tmp_path / "festival_interim.parquet"
    df_interim = build_festival_interim(
        raw_festival_dir=raw_dir,
        supplement_csv_path=supp_csv,
        output_path=interim_out,
    )

    assert interim_out.exists()
    # 유효 이벤트는 해운대모래축제(1), 2019 금정산성축제(1), 부산국제보트쇼(1), 지스타 2024(1) 총 4건이어야 함
    assert len(df_interim) == 4

    names = set(df_interim["event_name"].to_list())
    assert "제18회 해운대모래축제" in names
    assert "2019 금정산성축제" in names
    assert "부산국제보트쇼" in names
    assert "지스타 2024 (G-STAR)" in names
    assert "제11회 수영강변축제" not in names  # 차년도 1일 플레이스홀더 필터링 확인
    assert "소규모 개인 초대전" not in names  # 개인 화랑 필터링 확인
    assert "박물관 연간 상설전시" not in names  # 상설 전시 필터링 확인

    geumjeong_row = df_interim.filter(
        pl.col("event_name") == "2019 금정산성축제"
    ).to_dicts()[0]
    assert geumjeong_row["lat"] == pytest.approx(35.2453, abs=1e-4)
    assert geumjeong_row["lng"] == pytest.approx(129.0635, abs=1e-4)

    # 지스타 2024의 일평균 방문객(215,000 / 4 = 53,750) 및 scale_weight 검증
    gstar_row = df_interim.filter(
        pl.col("event_name") == "지스타 2024 (G-STAR)"
    ).to_dicts()[0]
    assert gstar_row["duration_days"] == 4
    assert gstar_row["daily_visitors"] == pytest.approx(53750.0)
    assert gstar_row["scale_weight"] == pytest.approx(
        math.log10(1.0 + 53750.0), rel=1e-5
    )


def test_build_festival_processed_rank_and_distance_weights(tmp_path: Path) -> None:
    """
    2단계(build_festival_processed)가 역사별 크기형 순위(nearest_event_rank: 3, 2, 1)와
    연속형 거리 가중치(max_dist_weight) 및 복합 충격량(event_impact_score)을 산출하는지 검증합니다.
    """
    station_master_path = REFERENCE_DATA_DIR / "station_master.parquet"
    assert station_master_path.exists()

    # 벡스코 단일 이벤트 (2024-11-14 ~ 2024-11-15, 2일간)
    scale_w = round(math.log10(1.0 + 50000.0), 6)
    df_interim = pl.DataFrame(
        {
            "event_name": ["지스타 2024"],
            "category": ["exhibition"],
            "venue_name": ["벡스코"],
            "start_date": [date(2024, 11, 14)],
            "end_date": [date(2024, 11, 15)],
            "duration_days": [2],
            "lat": [35.1691],
            "lng": [129.1362],
            "total_visitors": [100000],
            "daily_visitors": [50000.0],
            "scale_weight": [scale_w],
            "trfc_info": ["센텀시티역"],
            "source": ["test"],
        }
    )

    proc_out = tmp_path / "festival_processed.parquet"
    df_proc = build_festival_processed(
        interim_df=df_interim,
        station_master_path=station_master_path,
        output_path=proc_out,
        max_radius_km=1.5,
    )

    assert proc_out.exists()
    assert len(df_proc) > 0

    # 센텀시티역(station_id=206)은 1순위 최인접역이므로 nearest_event_rank == 3 이어야 함
    centum_rows = df_proc.filter(pl.col("station_id") == 206).sort("date")
    assert len(centum_rows) == 2  # 2일간 전개 확인
    assert centum_rows["nearest_event_rank"].to_list() == [3, 3]
    assert centum_rows["is_festival"].to_list() == [1, 1]

    c_row = centum_rows.to_dicts()[0]
    expected_dist_w = 1.0 / (1.0 + c_row["min_event_dist_km"])
    assert c_row["max_dist_weight"] == pytest.approx(expected_dist_w, abs=1e-4)
    assert c_row["event_impact_score"] == pytest.approx(
        scale_w * c_row["max_dist_weight"], abs=1e-3
    )

    # 시립미술관역(station_id=205)은 2순위 인접역이므로 nearest_event_rank == 2 이어야 함
    museum_rows = df_proc.filter(pl.col("station_id") == 205).sort("date")
    assert museum_rows["nearest_event_rank"].to_list() == [2, 2]

    # 반경 내 1, 2, 3순위 역이 모두 존재하는지 확인
    ranks_present = set(df_proc["nearest_event_rank"].to_list())
    assert ranks_present == {1, 2, 3}


def test_process_festival_data_e2e_with_tmp_dir(tmp_path: Path) -> None:
    """
    통합 진입점 process_festival_data가 major_events_supplement.csv 및 station_master.parquet를 바탕으로
    결측치 없이 주요 대형 전시·축제(지스타, 모빌리티쇼, 불꽃축제, BIFF, 록페스티벌, 바다축제)의
    정확한 일자·위경도·역사 매칭 피처를 생성하는지 검증합니다.
    """
    raw_dir = tmp_path / "raw_empty"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_out = tmp_path / "interim" / "festival_interim.parquet"
    proc_out = tmp_path / "processed" / "festival_processed.parquet"

    df_proc = process_festival_data(
        file_path=raw_dir,
        station_master_path=REFERENCE_DATA_DIR / "station_master.parquet",
        interim_output_path=interim_out,
        processed_output_path=proc_out,
    )
    df_interim = pl.read_parquet(interim_out)

    assert interim_out.exists()
    assert proc_out.exists()
    assert len(df_interim) >= 50
    assert len(df_proc) > 0

    # 1. 결측치(Null) 전수 검사
    assert sum(df_interim.null_count().row(0)) == 0
    assert sum(df_proc.null_count().row(0)) == 0

    # 2. 일자 역전(start_date > end_date) 및 위경도 도메인 범위 검사
    assert len(df_interim.filter(pl.col("start_date") > pl.col("end_date"))) == 0
    assert df_interim["lat"].min() >= 34.8
    assert df_interim["lat"].max() <= 35.5
    assert df_interim["lng"].min() >= 128.7
    assert df_interim["lng"].max() <= 129.4
    assert df_interim["total_visitors"].min() > 0
    assert df_interim["scale_weight"].min() > 0.0

    # 3. 주요 대형 전시·축제의 역사별 시공간 매칭(nearest_event_rank) 검증
    expected_mappings = [
        # (일자, 역사 ID, 기대 nearest_event_rank)
        (date(2024, 11, 14), 205, 3),  # 지스타 2024 -> 205 시립미술관(벡스코) 1순위
        (date(2024, 11, 14), 206, 2),  # 지스타 2024 -> 206 센텀시티 2순위
        (date(2024, 11, 14), 207, 1),  # 지스타 2024 -> 207 민락 3순위
        (
            date(2024, 6, 29),
            205,
            3,
        ),  # 2024 부산모빌리티쇼 -> 205 시립미술관(벡스코) 1순위
        (date(2024, 6, 29), 206, 2),  # 2024 부산모빌리티쇼 -> 206 센텀시티 2순위
        (date(2024, 11, 9), 209, 3),  # 제19회 부산불꽃축제 -> 209 광안 1순위
        (date(2024, 11, 9), 210, 2),  # 제19회 부산불꽃축제 -> 210 금련산 2순위
        (date(2024, 10, 5), 206, 3),  # 제29회 부산국제영화제 -> 206 센텀시티 1순위
        (date(2024, 10, 5), 227, 3),  # 2024 부산국제록페스티벌 -> 227 사상 1순위
        (date(2024, 7, 26), 95, 3),  # 제28회 부산바다축제 -> 95 다대포해수욕장 1순위
    ]
    for target_date, sid, expected_rank in expected_mappings:
        matched = df_proc.filter(
            (pl.col("date") == target_date) & (pl.col("station_id") == sid)
        )
        assert len(matched) == 1, f"date={target_date}, station_id={sid} 매칭 행 누락"
        row = matched.to_dicts()[0]
        assert row["nearest_event_rank"] == expected_rank
        assert row["max_dist_weight"] > 0.0
        assert row["event_impact_score"] > 0.0


def test_cancelled_event_filtering_and_same_year_dedup(tmp_path: Path) -> None:
    """
    취소/잠정연기 행사가 제외되고, 동일 연도·동일 행사가 여러 원천에 중복 존재할 때
    우선순위(major_events_supplement.csv > KC_488)에 따라 단일화되는지 검증합니다.
    """
    raw_dir = tmp_path / "raw_festival"
    kc_dir = raw_dir / "전국 문화축제 표준데이터(시군구 행정동 100미터격자코드 포함)"
    kc_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "FCLTY_NM": [
                "제9회 낙동강 구포나루 축제(잠정연기)",
                "2022 낙동강유채꽃축제",
                "제19회 부산불꽃축제",
            ],
            "CTPRVN_NM": ["부산광역시", "부산광역시", "부산광역시"],
            "OPMTN_PLACE_NM": ["화명생태공원", "대저생태공원", "광안리해수욕장"],
            "FSTVL_CN": ["감동 민속촌", "축제취소, 자율관람", "불꽃쇼"],
            "FSTVL_BEGIN_DE": ["2020-05-17", "2022-04-09", "2024-11-02"],
            "FSTVL_END_DE": ["2020-05-19", "2022-04-17", "2024-11-02"],
            "FCLTY_LA": [35.1968, 35.2098, 35.1532],
            "FCLTY_LO": [128.9900, 128.9845, 129.1186],
        }
    ).write_csv(kc_dir / "KC_488_WNTY_CLTFSTVL_2024.csv")

    supp_csv = tmp_path / "supplement.csv"
    pl.DataFrame(
        {
            "event_name": ["제19회 부산불꽃축제"],
            "category": ["festival"],
            "venue_name": ["광안리해수욕장"],
            "start_date": ["2024-11-09"],
            "end_date": ["2024-11-09"],
            "lat": [35.1532],
            "lng": [129.1186],
            "total_visitors": [1030000],
            "trfc_info": ["광안역 금련산역"],
            "source_note": ["부산광역시"],
        }
    ).write_csv(supp_csv)

    interim_out = tmp_path / "festival_interim.parquet"
    df_interim = build_festival_interim(
        raw_festival_dir=raw_dir,
        supplement_csv_path=supp_csv,
        output_path=interim_out,
    )

    # 취소·연기 2건은 제외되고, 부산불꽃축제는 supplement.csv 기준(2024-11-09) 1건만 남아야 함
    assert len(df_interim) == 1
    row = df_interim.to_dicts()[0]
    assert row["event_name"] == "제19회 부산불꽃축제"
    assert row["start_date"] == date(2024, 11, 9)
    assert row["total_visitors"] == 1030000
    assert row["source"] == "major_events_supplement.csv"


def test_build_festival_from_samples_dir(tmp_path: Path) -> None:
    """
    CI 환경에서 data/samples/festival/ 에 적재된 5개 샘플 파일(전체 스키마 보존 및 민감정보 제거본)을 바탕으로
    interim 및 processed 2단계 파이프라인이 정상 구동되고 모든 엣지 케이스가 필터링/교정되는지 검증합니다.
    """
    sample_fest_dir = SAMPLES_DIR / "festival"
    assert sample_fest_dir.exists(), "data/samples/festival 디렉터리가 존재해야 합니다."

    interim_out = tmp_path / "samples_interim.parquet"
    proc_out = tmp_path / "samples_processed.parquet"

    df_proc = process_festival_data(
        file_path=sample_fest_dir,
        station_master_path=REFERENCE_DATA_DIR / "station_master.parquet",
        interim_output_path=interim_out,
        processed_output_path=proc_out,
    )
    df_interim = pl.read_parquet(interim_out)

    assert interim_out.exists()
    assert proc_out.exists()
    assert len(df_interim) >= 60  # major_events_supplement 58 + samples 유효 레코드
    assert len(df_proc) > 0

    # 1. 결측치 0건 전수 검사
    assert sum(df_interim.null_count().row(0)) == 0
    assert sum(df_proc.null_count().row(0)) == 0

    # 2. samples 내 유효 이벤트 포함 및 좌표·오타 교정 값 검증
    names = set(df_interim["event_name"].to_list())
    assert "제18회 해운대모래축제" in names
    assert "2019 금정산성축제" in names
    assert "광안리어방축제" in names
    assert "2024 부산국제보트쇼" in names
    assert "부산자갈치축제" in names
    assert "벡스코 특별기획전" in names

    # 위경도 반전(129.x, 35.x -> 35.x, 129.x) 실제 교정 확인
    geumjeong = df_interim.filter(
        pl.col("event_name") == "2019 금정산성축제"
    ).to_dicts()[0]
    assert 35.0 <= geumjeong["lat"] <= 35.5
    assert 128.7 <= geumjeong["lng"] <= 129.4

    # 종료일 오타(10.06 -> 10.26) 실제 교정 확인
    jagalchi = df_interim.filter(pl.col("event_name") == "부산자갈치축제").to_dicts()[0]
    assert jagalchi["end_date"] >= jagalchi["start_date"]
    assert jagalchi["end_date"].month == 10
    assert jagalchi["end_date"].day == 26

    # 3. samples 내 필터링 대상 제외 여부 확인
    assert "제11회 수영강변축제" not in names  # 차년도 1일 플레이스홀더 제외
    assert "2022 낙동강유채꽃축제" not in names  # 축제취소 제외
    assert "제9회 낙동강 구포나루 축제(잠정연기)" not in names  # 잠정연기 제외
    assert "서울빛초롱축제" not in names  # 타 시도(서울) 제외
    assert "진주남강유등축제" not in names  # 타 시도(경남) 제외
    assert "2024 서울모빌리티쇼" not in names  # 킨텍스 제외
    assert "소규모 개인 초대전" not in names  # 개인 화랑 제외
    assert "박물관 연간 상설전시" not in names  # 120일 초과 상설전시 제외
