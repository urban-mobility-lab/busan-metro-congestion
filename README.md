# 부산 지하철 혼잡도 예측 플랫폼 (Busan Metro Congestion Platform)

> 외부 요인(기상, 축제, 공휴일)을 반영한 머신러닝/딥러닝 기반 부산 지하철 혼잡도 예측 시스템

---

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 부산 공공데이터(역별/시간대별 승하차 인원, 축제·전시 일정), 기상청 단기예보 및 한국천문연구원 특일 정보(공휴일) 데이터를 결합하여, 기상 악화·공휴일·대규모 축제 등 비정기적 외부 요인에 대응하는 부산 지하철 혼잡도 예측 플랫폼을 구축합니다.

---

## 2. 빠른 시작 가이드 (Quick Start)

### 2.1 로컬 개발 환경 구성 (`uv` 기반)
로컬 개발 환경에서는 Docker 없이 `uv`를 통해 개발 및 검증을 진행할 수 있습니다.

```bash
# 1. uv를 통한 가상환경 생성 및 의존성 설치
uv sync --extra dev

# 2. 공공데이터 API 키 설정
cp .env.template .env
# .env 파일을 열어 data.go.kr에서 발급받은 PUBLIC_DATA_PORTAL_KEY를 입력합니다.

# 3. 코드 스타일, 린트 및 단위 테스트 검증
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
```

### 2.2 공공데이터 수집 및 참조 마스터 구축 (`scripts/`)
원천 API 수집과 참조 마스터 테이블 생성은 `scripts/` 하위 CLI 스크립트로 실행합니다.

```bash
# 1. 공공데이터 OpenAPI 원천 수집 (.env의 PUBLIC_DATA_PORTAL_KEY 필요)
uv run python scripts/fetch_weather_forecast_api.py
uv run python scripts/fetch_busan_events_api.py
uv run python scripts/fetch_calendar_holidays_api.py

# 2. 참조 마스터 테이블 생성 (data/reference/*.parquet)
uv run python scripts/build_station_master.py
uv run python scripts/build_station_weather_grid.py
uv run python scripts/build_calendar_master.py
```

### 2.3 운영 파이프라인 실행 (`busan-metro`)

E2E 파이프라인은 모델 학습 및 평가를 수행하는 `train` 서브커맨드와 추론 서빙을 위한 `predict` 서브커맨드를 제공합니다 (서브커맨드 생략 시 기본 `train` 모드로 동작합니다).

#### 모델 학습 및 평가 (`train`)
```bash
# 1. 전체 데이터 대상 배치 학습 및 2-Track 앙상블 평가
uv run busan-metro train

# 2. 사용자 지정 1년 단위 시계열 구간 분할 실행 (종료일 생략 시 시작일 + 1년 자동 계산)
uv run busan-metro train --train-start 2023-06-01 --val-start 2024-06-01 --test-start 2025-06-01

# 3. 전처리 단계만 실행 (--step preprocess | train_eval | all)
uv run busan-metro train --step preprocess
```

#### 추론 서빙 인터페이스 (`predict`)
```bash
# 기학습 모델 기반 실시간 추론 실행
uv run busan-metro predict --horizon short
```

### 2.4 연구 파이프라인 실행 (Optuna 2-Track Ablation)
5대 Feature Group(Set 0~4) 대상 Optuna 하이퍼파라미터 튜닝, 2-Track 롤링·홀드아웃 평가 및 논문 표 생성을 수행합니다.

```bash
# 1. 전체 Optuna 튜닝 및 2-Track 롤링·홀드아웃 평가 실행
uv run python scripts/run_optuna_2track_ablation.py

# 2. 저장된 최적 파라미터로 튜닝 생략 후 평가만 재실행
uv run python scripts/run_optuna_2track_ablation.py --skip-tuning

# 3. 저장된 체크포인트 가중치 기반 추론 평가 / 결과 표 마크다운만 재생성
uv run python scripts/run_optuna_2track_ablation.py --eval-from-weights
uv run python scripts/run_optuna_2track_ablation.py --tables-only
```

### 2.5 GPU 배치 환경 실행 (Docker Compose)
Docker 환경에서 격리 실행할 때 사용합니다:
```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## 3. 프로젝트 문서 링크

* [DIRECTORY_STRUCTURE.md](./docs/DIRECTORY_STRUCTURE.md): 디렉터리 구조 및 모듈 매핑 명세
* [API_SPEC.md](./docs/API_SPEC.md): 사전 추론 예측 결과 조회 REST API 명세
* [CONTRIBUTING.md](./CONTRIBUTING.md): 협업 브랜치 전략, 커밋 컨벤션, PR 규칙
* [AGENTS.md](./AGENTS.md): AI 에이전트를 위한 개발 가이드라인
