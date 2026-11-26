# 웹: 시간대별 이용 예측 조회

빌드 없는 정적 페이지입니다. 호선 → 역 → 날짜를 고르면 `GET /api/v1/predictions` 결과를 그래프와 표로 보여줍니다.

- 막대는 `predicted_passengers`, 선은 `typical_passengers`(같은 역·요일·시간대의 과거 대표값)입니다.
- 값은 역의 시간당 승차+하차 합계 인원이며 혼잡률(%)이 아닙니다.
- 가장 붐비는 시간은 빨간 막대, 가장 한산한 시간은 초록 막대로 강조합니다. 한산한 시간은 운행 시간대(예측값과 과거 대표값이 모두 0보다 큰 시간) 중에서 고릅니다.
- `hour=24`는 선택한 날짜 다음 날 00~01시로 표시합니다.
- `is_template`이 `true`이면 결과 위에 임의값 경고를 띄웁니다.
- 오류 응답은 `code`별 안내 문구(`app.js`의 `ERROR_TEXT`)로 보여줍니다.

## 경로 조회 탭

출발역 → 도착역 → 날짜 → 출발 시각을 고르면, 환승 2회까지 가장 빨리 도착하는 경로를 찾아 각 역 도착 시각과 그 시간대의 `predicted_passengers`·`typical_passengers`를 역마다 보여줍니다.

- 도착 시각과 환승 도보 시간은 부산교통공사 열차시각표·역간 소요시간(표준 시각)입니다. `web/data/timetable.js`에 들어 있으며 아래 순서로 만듭니다.
  ```bash
  uv run python scripts/fetch_train_timetable_api.py
  uv run python scripts/export_web_timetable.py
  ```
- 평일·토요일·일요일 시각표를 날짜로 고르고, 공휴일은 일요일 시각표를 씁니다.
- 환승역(서면, 연산, 덕천, 수영, 미남, 동래)은 역 이름 하나로 고르고, 계산할 때 빠른 호선을 고릅니다. 환승역에서는 도보 시간이 지난 뒤 처음 오는 열차를 탑니다. 계산 규칙은 `busan_metro_congestion/service/route_timeline.py`의 `plan_journey`와 같습니다.
- 인원은 역 전체의 시간당 승하차 인원이며 열차 안 혼잡률이 아닙니다. 경로 중 가장 큰 역을 빨간색으로 강조합니다.

## 실행

```bash
uv run python -m http.server 8090 -d web
```

브라우저에서 `http://localhost:8090`을 엽니다. 이렇게 띄우면 API가 없으므로 목업으로 동작합니다. 외부 라이브러리는 Chart.js 4.4.1(MIT) 하나입니다. 인터넷 없이도 동작하도록 `vendor/chart.umd.js`에 사본을 두고, 이 파일이 없을 때만 jsDelivr CDN(버전 고정)에서 받습니다.

## 목업과 실제 API 전환

기본값(`config.js`의 `useMock: "auto"`)은 페이지를 열 때 `/api/v1/health`를 불러 봅니다. 응답하면 실제 API를 쓰고, 응답하지 않으면 목업으로 동작합니다. 그래서 API 서버가 `web/`을 함께 제공하면 설정을 바꾸지 않아도 실제 API로 동작합니다.

| 방법 | 설정 |
| --- | --- |
| 자동 (기본값) | `useMock: "auto"`, `apiBaseUrl: ""` |
| 항상 목업 | `useMock: true` 또는 주소 뒤에 `?mock=1` |
| 다른 서버의 API | `useMock: false`, `apiBaseUrl: "http://localhost:8000"` 또는 주소 뒤에 `?api=http://localhost:8000` |

API를 다른 출처(주소·포트)에서 호출하면 API 서버에 CORS 허용이 필요합니다. API 서버가 `web/`을 같은 출처로 제공하면 필요 없습니다.

오류 상태 코드는 B 전달 문서를 따릅니다: `STATION_NOT_FOUND` 404, `DATE_OUT_OF_RANGE`·`INVALID_HOUR`·`INVALID_REQUEST` 422, `PREDICTION_NOT_READY` 503. 화면은 상태 코드보다 응답의 `code`를 먼저 보고, 서버가 준 한국어 `message`도 함께 보여 줍니다. `model-info`에 `prediction_start`, `prediction_end`가 없으면 날짜 제한 없이 내일을 기본값으로 둡니다.

목업 모드에서는 화면 위쪽 "응답 시나리오"에서 template 데이터, 각 오류 `code`, HTTP 500, 네트워크 실패, 느린 응답(로딩 표시)을 재현할 수 있습니다. 예측 가능 기간 밖의 날짜를 직접 입력해도 `DATE_OUT_OF_RANGE`가 납니다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `index.html` | 화면 구조 |
| `style.css` | 스타일 |
| `config.js` | 목업/실제 API 전환 설정 |
| `app.js` | API 호출, 응답 정규화(`normalizeStations`, `normalizePrediction`), 그래프·표 렌더링 |
| `route.js` | 경로 조회 탭: 환승 경로 탐색, 역별 도착 시각 계산, 역별 예측 조회, 타임라인 렌더링 |
| `data/timetable.js` | `scripts/export_web_timetable.py`가 만든 열차시각표와 환승 도보 시간(직접 수정 금지) |
| `vendor/chart.umd.js` | Chart.js 4.4.1 로컬 사본 |
| `mock/stations.js` | `data/reference/station_master.parquet`에서 뽑은 역 목록 |
| `mock/mock-api.js` | API 초안 형식의 목업 응답 생성 |

API 응답 형식이 초안과 달라지면 `app.js`의 `normalizeStations`, `normalizePrediction`만 맞추면 됩니다.

## 목업이 가정한 응답 형식

```json
{
  "station": {"station_id": 119, "station_name": "서면", "line_num": 1},
  "date": "2026-11-13",
  "model_version": "mock-day-ahead-0.1",
  "is_template": false,
  "generated_at": "2026-11-12T17:00:00+09:00",
  "predictions": [
    {"hour": 1, "predicted_passengers": 0, "typical_passengers": 0},
    {"hour": 8, "predicted_passengers": 4321, "typical_passengers": 4480}
  ]
}
```

`GET /api/v1/stations`는 역 객체 배열(또는 `{"stations": [...]}`), `GET /api/v1/model-info`는 `model_version`, `prediction_start`, `prediction_end`를 가진 객체로 가정합니다.
