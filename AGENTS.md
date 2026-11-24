# AGENTS.md

부산 1~4호선 승하차 데이터와 외부 요인(기상·축제·공휴일)으로 역사별 시간대 승하차 통행량(`accumulated_passengers`)을 예측하는 MLOps 파이프라인.
디렉터리 구조는 `docs/DIRECTORY_STRUCTURE.md`를 참고한다.

## 파일 위치 및 데이터 규칙

- 테스트는 Git 추적 데이터(`data/samples/`, `data/reference/`)만 사용한다. `data/raw/`, `data/interim/`, `data/features/`, `data/processed/`는 Git에서 제외되어 CI에 없다.
- `data/raw/`, `data/processed/` 등 원천·전처리 파일 부재 시 `data/samples/`로 암묵적 폴백하지 않고 `FileNotFoundError`로 명시 실패한다(테스트에서만 경로 인자로 명시 주입).
- 룩업·마스터 참조 테이블은 `data/reference/`를 사용한다. OpenAPI 활용가이드 문서는 `data/raw/` 각 도메인 하위 폴더에 있다(로컬 전용, Git 미추적).
- 가중치·스케일러·Optuna DB 등 체크포인트 산출물은 `artifacts/**/checkpoints/`에 격리(Git 미추적)하며, 평가 리포트(JSON/Markdown)는 각 실험 폴더(`artifacts/baseline/`, `artifacts/ablation/`)에 저장한다.
- 일회성 스크립트는 `scratch/`에 둔다.
- `data/` 하위 파일의 원천 인코딩(CP949/UTF-8 BOM/UTF-16)과 줄바꿈은 임의 변환하지 않는다(`.gitattributes` 준수). `data/samples/` 편집 시 개인정보와 서술형 본문을 배제한다(`data/samples/README.md` 준수).

## 아키텍처 및 도메인 불변식

- 의존성 방향: `scripts/`(수집·빌더·Ablation 연구)는 `busan_metro_congestion/`을 import할 수 있지만, `busan_metro_congestion/`은 `scripts/`를 import하지 않는다.
- 2-Track 호라이즌: `short`(<=12h, 기상 포함 25종)와 `long`(>12h, Weather-Free 17종)을 분리 유지한다. `short` 입력에 `lead_time_hours > 12` 또는 기상 결측의 임의 0 대치를 금지한다(`validate_short_horizon_lead_time` 준수).
- Random Forest는 대용량 추론 SIMD hang 방지를 위해 scikit-learn `RandomForestRegressor` 직접 사용 대신 `SafeRandomForestRegressor`를 사용한다.
- 시계열 누출 방지·경계 격리·재현성: 무작위 셔플 분할 금지, 스케일러·타깃 인코더는 학습셋 기준 적합, `MetroSlidingWindowDataset` 생성 시 `station_ids` 전달로 역사 경계 혼합 차단, 학습·실험 진입점에서 `busan_metro_congestion.config.set_seed()` 호출.

## 금지

- pandas를 import하지 않는다. Polars/PyArrow를 사용한다.
- 절대 경로를 쓰지 않는다. `busan_metro_congestion.config.PROJECT_ROOT` 또는 `Path(__file__).resolve()` 기준으로 구성한다.
- `.to("cuda")`를 하드코딩하지 않는다. `busan_metro_congestion.config.DEVICE`를 사용한다.
- `.env`(`PUBLIC_DATA_PORTAL_KEY`)를 커밋·하드코딩·로그 출력하지 않는다. API 키는 `busan_metro_congestion.config.get_api_key()`로 로드한다.
- 실패하는 테스트를 `skip`/`xfail`로 우회하지 않는다.
- 항진 명제·상수 재확인·구현 우회 모킹 등 형식적 테스트를 작성하지 않는다.
- 변경 이력성 주석("기존에는 ~", "향후 구현 예정")을 문서/코드에 남기지 않는다.

## 사전 승인 필요

- 핵심 모듈·컨테이너 수정: `busan_metro_congestion/config.py`, `busan_metro_congestion/run_pipeline.py`, `docker/`
- 의존성·빌드·CI 설정 변경: `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`

## 테스트 및 검증

- 기능 코드를 추가하거나 수정하면 핵심 불변식·경계 조건·실패 경로를 검증하는 테스트를 같은 작업에서 함께 작성한다.
- 단위 테스트에서 외부 OpenAPI 실호출을 금지한다(오프라인 모킹 및 Git 추적 데이터만 사용).
- 문서(`*.md`)만 수정한 경우를 제외하고 완료 전 `uv run ruff format .`, `uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest tests/ -q`를 통과시킨다.

## 기여 규칙

브랜치 생성, 커밋 메시지, PR 절차는 `CONTRIBUTING.md`를 따른다. 커밋이나 PR을 만들기 전에 읽는다.

## 명령어

```bash
uv sync --extra dev                         # 의존성 설치 (개발 도구 포함)
uv run pytest tests/ -v                     # 전체 테스트
uv run pytest tests/<file>.py::<name> -x    # 단일 테스트
uv run ruff check .                         # 린트 검사
uv run ruff format .                        # 코드 포맷팅
uv run ruff format --check .                # 포맷 검사 (CI 게이트와 동일)
uv run busan-metro                          # E2E 파이프라인 실행 (기본 train 모드)
```
