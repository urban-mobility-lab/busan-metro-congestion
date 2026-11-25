# 사전 추론 혼잡도 예측 결과 조회 API 명세서 (API_SPEC.md)

> 관련 이슈: #27 `[feat] 사전 추론 혼잡도 예측 결과 조회 API 구현`
>
> 배치 추론 파이프라인(`busan_metro_congestion/modeling/predict.py`)이 사전 적재한 예측 결과 저장소를 조회해 노선·역사·날짜·시간대별 예측 승객 수를 반환하는 경량 REST API의 명세입니다. WebUI와 자연어 AUI 프롬프트 Skill이 공통으로 호출하는 표준 응답 스키마를 정의합니다.

---

## 1. 개요

| 항목 | 내용 |
| --- | --- |
| 프레임워크 | FastAPI (OpenAPI 3 자동 생성) |
| 패키지 위치 | `busan_metro_congestion/api/` |
| Base Path | `/api/v1` |
| 실행 | `uv run uvicorn busan_metro_congestion.api.app:app` |
| 대화형 문서 | `GET /docs` (Swagger UI), `GET /openapi.json` |
| 인증 | 없음 (조회 전용 공개 API) |
| 지원 메서드 | `GET` 전용 (그 외 메서드는 `405`) |
| 문자 인코딩 | `application/json; charset=utf-8` |
| 시간대 | 모든 날짜·시각은 KST(UTC+09:00) 기준 |

### 1.1 모듈 구성

| 파일 | 역할 |
| --- | --- |
| `api/app.py` | `create_app()` 앱 팩토리, `/api/v1` 라우트 정의 |
| `api/predictions.py` | 예측 결과 저장소(Parquet) 로드·스키마 검증·조회 |
| `api/stations.py` | `data/reference/station_master.parquet` 기반 역 목록 |
| `api/schemas.py` | 요청/응답 Pydantic 스키마 |
| `api/errors.py` | `{"code", "message"}` 형식 오류 응답 통일 |

`create_app()`은 저장소 경로와 역 마스터 경로를 인자로 받아, 테스트에서 `data/samples/predictions/predictions_sample.parquet`를 주입해 오프라인으로 검증합니다.

---

## 2. 데이터 소스 계약

### 2.1 예측 결과 저장소

| 항목 | 내용 |
| --- | --- |
| 운영 경로 | `artifacts/predictions/predictions.parquet` (`modeling.predict.DEFAULT_PREDICTIONS_PATH`) |
| 테스트 샘플 | `data/samples/predictions/predictions_sample.parquet` (4개 역 × 2일 × 24시간 = 192행) |
| 생성 주체 | `run_batch_inference()` — 임시 파일 + `os.replace()` 원자적 교체 |
| 복합 키 | `(station_id, target_date, target_hour)` 유일 |

API는 배치 추론 산출물 스키마 계약 `PREDICTION_SCHEMA`(9컬럼)를 그대로 따르며, 로드 시 `modeling.predict.validate_predictions_df()`로 정합성을 검증합니다.

| 컬럼 | Polars 타입 | 설명 | 응답 필드 |
| --- | --- | --- | --- |
| `station_id` | `Int64` | 역 번호 | `station_id` |
| `station_name` | `String` | 역 이름 | `station_name` |
| `line_num` | `Int8` | 호선 (1~4) | `line_num` |
| `target_date` | `Date` | 예측 대상 운행일 | `date` |
| `target_hour` | `Int8` | 시간대 (1~24) | `predictions[].hour` |
| `target_datetime` | `Datetime("us")` | 시간대 물리 시각 (24시 → 익일 00:00) | `predictions[].target_datetime` |
| `predicted_passengers` | `Float32` | 예측 승객 수 (≥ 0, 유한값) | `predictions[].predicted_passengers` |
| `model_version` | `String` | 모델 버전 | `model_version` |
| `generated_at` | `Datetime("us")` | 배치 생성 시각 (naive UTC) | `generated_at` (KST 변환) |

### 2.2 시간대 규약

- `hour`(= `target_hour`)는 1~24 정수입니다. `8`은 08:00~09:00 구간, `24`는 다음 날 00:00~01:00 구간입니다.
- `24`시는 `date`의 운행일에 속하며, `target_datetime`은 익일 `00:00:00`입니다.
- `generated_at`은 저장소에 naive UTC로 기록되어 있으며, API는 `+09:00` 오프셋이 붙은 KST ISO 8601 문자열로 변환해 반환합니다.

### 2.3 역 마스터

- `data/reference/station_master.parquet` (1~4호선 114개 역)에서 `station_id`, `station_name`, `line_num`, `lat`, `lng`를 읽습니다.
- 역 존재 여부(`STATION_NOT_FOUND`)는 역 마스터 기준으로 판정합니다.

### 2.4 로드 및 갱신

- 저장소는 첫 요청 시 로드하고 메모리에 보관합니다.
- 요청마다 파일의 수정 시각(mtime)을 확인해, 배치 추론이 파일을 교체하면 다음 요청에서 다시 로드합니다.
- 파일 부재, 0행, 스키마 검증 실패 시 해당 요청은 `503 PREDICTION_NOT_READY`로 응답하며 실패 상태를 캐시하지 않습니다.

---

## 3. 공통 규칙

### 3.1 오류 응답 형식

모든 오류는 아래 형식으로 통일합니다.

```json
{
  "code": "STATION_NOT_FOUND",
  "message": "역 번호 999을(를) 찾을 수 없습니다."
}
```

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `code` | string | 기계 판독용 오류 코드 |
| `message` | string | 사용자에게 그대로 보여줄 수 있는 한국어 메시지 |

### 3.2 오류 코드

| HTTP | `code` | 발생 조건 |
| --- | --- | --- |
| 404 | `STATION_NOT_FOUND` | 역 마스터에 없는 `station_id` |
| 404 | `INVALID_REQUEST` | 정의되지 않은 경로 |
| 405 | `INVALID_REQUEST` | `GET` 이외 메서드 |
| 422 | `INVALID_REQUEST` | 필수 파라미터 누락, 타입·형식 오류 (예: `date=2026-13-01`) |
| 422 | `INVALID_HOUR` | `hour`가 1~24 범위 밖 |
| 422 | `INVALID_LINE` | `line_num`이 1~4 범위 밖 |
| 422 | `DATE_OUT_OF_RANGE` | 저장소에 해당 역·날짜의 예측이 없음 (예측 가능 기간 밖) |
| 503 | `PREDICTION_NOT_READY` | 저장소 파일 부재·빈 저장소·스키마 계약 위반, 처리 중 예기치 못한 오류 |

검증 순서는 `파라미터 형식 → hour/line_num 범위 → 역 존재 → 저장소 준비 → 날짜 범위`입니다.

---

## 4. 엔드포인트

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| GET | `/api/v1/health` | 서버 생존 확인 |
| GET | `/api/v1/stations` | 역 목록 (노선 필터) |
| GET | `/api/v1/model-info` | 저장소 메타데이터 (모델 버전, 예측 가능 기간) |
| GET | `/api/v1/predictions` | 역·날짜·시간대별 예측 승객 수 |

### 4.1 `GET /api/v1/health`

서버 프로세스 생존 여부만 확인하며 저장소 상태와 무관하게 `200`을 반환합니다. 저장소 준비 여부는 `/model-info`로 확인합니다.

**응답 200**

```json
{ "status": "ok" }
```

### 4.2 `GET /api/v1/stations`

역 마스터의 역 목록을 `station_id` 오름차순으로 반환합니다.

**Query Parameters**

| 이름 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `line_num` | integer (1~4) | N | 지정 시 해당 호선 역만 반환 |

**응답 200** — `Station[]`

```json
[
  {
    "station_id": 119,
    "station_name": "서면",
    "line_num": 1,
    "lat": 35.158282,
    "lng": 129.059556
  }
]
```

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `station_id` | integer | 역 번호 |
| `station_name` | string | 역 이름 |
| `line_num` | integer | 호선 번호 (1~4) |
| `lat` | number | 위도 (WGS84) |
| `lng` | number | 경도 (WGS84) |

**오류**: `422 INVALID_LINE`, `422 INVALID_REQUEST`

### 4.3 `GET /api/v1/model-info`

현재 적재된 예측 결과 저장소의 메타데이터를 반환합니다. 클라이언트는 `available_from`~`available_to`로 날짜 선택 범위를 제한합니다.

**응답 200**

```json
{
  "model_version": "0.1.0-set1-sample",
  "generated_at": "2026-11-25T19:00:00+09:00",
  "available_from": "2026-12-01",
  "available_to": "2026-12-05",
  "station_count": 4,
  "row_count": 192
}
```

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `model_version` | string | 저장소의 `model_version` |
| `generated_at` | string (date-time, KST) | 저장소의 `generated_at` 최댓값 |
| `available_from` | string (date) | `target_date` 최솟값 |
| `available_to` | string (date) | `target_date` 최댓값 |
| `station_count` | integer | 저장소에 포함된 고유 역 수 |
| `row_count` | integer | 저장소 전체 행 수 |

`available_from`~`available_to`는 저장소의 최소·최대 날짜이며, 그 사이 날짜가 모두 존재함을 보장하지는 않습니다. 개별 날짜의 존재 여부는 `/predictions` 응답으로 판정합니다.

**오류**: `503 PREDICTION_NOT_READY`

### 4.4 `GET /api/v1/predictions`

한 역의 하루 동안 시간대별 예측 승객 수를 반환합니다.

**Query Parameters**

| 이름 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `station_id` | integer | Y | 역 번호 |
| `date` | string (`YYYY-MM-DD`) | Y | 조회 운행일 (KST) |
| `hour` | integer (1~24), 반복 가능 | N | 조회 시간대. 생략 시 1~24 전체. 중복은 제거하고 오름차순 정렬 |

**요청 예시**

```http
GET /api/v1/predictions?station_id=119&date=2026-12-01&hour=8&hour=24
```

**응답 200** — `PredictionResponse`

```json
{
  "station_id": 119,
  "station_name": "서면",
  "line_num": 1,
  "lat": 35.158282,
  "lng": 129.059556,
  "date": "2026-12-01",
  "model_version": "0.1.0-set1-sample",
  "generated_at": "2026-11-25T19:00:00+09:00",
  "predictions": [
    {
      "hour": 8,
      "target_datetime": "2026-12-01T08:00:00+09:00",
      "predicted_passengers": 4195.6
    },
    {
      "hour": 24,
      "target_datetime": "2026-12-02T00:00:00+09:00",
      "predicted_passengers": 168.0
    }
  ]
}
```

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `station_id` ~ `lng` | — | `Station`과 동일 (역 마스터 기준) |
| `date` | string (date) | 조회 운행일 |
| `model_version` | string | 해당 행들의 `model_version` |
| `generated_at` | string (date-time, KST) | 해당 행들의 배치 생성 시각 |
| `predictions` | `HourlyPrediction[]` | 요청 시간대 순서의 예측 목록 |

`HourlyPrediction`

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `hour` | integer (1~24) | 시간대 |
| `target_datetime` | string (date-time, KST) | 시간대 시작 시각 (24시는 익일 00:00) |
| `predicted_passengers` | number | 예측 승객 수 (≥ 0, 소수 첫째 자리 반올림) |

**처리 규칙**

- 저장소에 `(station_id, date)` 행이 하나도 없으면 `422 DATE_OUT_OF_RANGE`를 반환합니다.
- 요청한 시간대 중 일부가 저장소에 없으면 부분 응답하지 않고 `503 PREDICTION_NOT_READY`를 반환합니다. 정상 산출물은 역·날짜마다 24개 시간대를 모두 포함합니다.

**오류**

| 상황 | 응답 |
| --- | --- |
| `station_id` 또는 `date` 누락 | `422 INVALID_REQUEST` — `"요청 값이 올바르지 않습니다. date 값이 필요합니다."` |
| `date=2026-12-32` | `422 INVALID_REQUEST` — `"요청 값이 올바르지 않습니다. date 값의 형식이 올바르지 않습니다."` |
| `hour=0` | `422 INVALID_HOUR` — `"hour는 1부터 24 사이여야 합니다. 받은 값: [0]"` |
| `station_id=999` | `404 STATION_NOT_FOUND` — `"역 번호 999을(를) 찾을 수 없습니다."` |
| `date=2027-01-01` | `422 DATE_OUT_OF_RANGE` — `"2027-01-01은(는) 예측 가능 기간(2026-12-01 ~ 2026-12-05)을 벗어났습니다."` |
| 저장소 부재·빈 저장소 | `503 PREDICTION_NOT_READY` — `"예측 모델이 준비되지 않았습니다. 잠시 후 다시 시도해 주세요."` |

---

## 5. 테스트 범위 (`tests/test_api.py`)

모든 테스트는 `data/samples/predictions/predictions_sample.parquet`와 `data/reference/station_master.parquet`만 사용하며 FastAPI `TestClient`로 오프라인 실행합니다.

| 분류 | 케이스 |
| --- | --- |
| 정상 조회 | `/health`, `/stations` 전체·`line_num` 필터, `/model-info` 메타데이터, `/predictions` 전체 24시간·`hour` 부분 조회·중복 `hour` 정렬 |
| 스키마 정합성 | 응답 필드가 `PREDICTION_SCHEMA`와 1:1 대응, 24시 `target_datetime` 익일 00:00, `generated_at` KST 변환 |
| 경계 조건 | 미존재 역(`404`), 저장소 기간 밖 날짜(`422`), `hour` 0·25(`422`), `line_num` 5(`422`), 필수 파라미터 누락·형식 오류(`422`) |
| 실패 경로 | 저장소 파일 부재·0행·컬럼 불일치(`503`), 실패 후 파일 생성 시 정상 복구, 파일 교체 시 재로드 |
| 공통 | 미정의 경로 `404`, `POST` `405`, 모든 오류 본문이 `{"code", "message"}` |
