# 프로젝트 디렉터리 구조 및 모듈 명세 (DIRECTORY_STRUCTURE.md)

> 설계 원칙: 폴더 깊이(depth)를 최대 2단계로 제한하며, 역할이 모호한 공통 폴더를 지양하고 직관적인 도메인 분리를 준수합니다.

---

```text
busan-metro-congestion/
│
├── .github/
│   ├── ISSUE_TEMPLATE/            # GitHub Issue 템플릿 (feat.md, fix.md, docs_chore.md)
│   ├── workflows/
│   │   └── ci.yml                 # GitHub Actions CI 워크플로우 (uv 기반 린트/포맷 및 pytest 테스트 검증)
│   └── pull_request_template.md   # GitHub PR 체크리스트 템플릿
│
├── .dockerignore                  # Docker 이미지 빌드 제외 대상 정의
├── .env.template                  # 공공데이터포털 API 키 환경변수 템플릿
├── .gitattributes                 # 줄바꿈(LF) 통일 및 데이터·바이너리 파일 보존 규칙
├── .gitignore                     # 대용량 데이터, 모델 가중치, 캐시 격리 및 추적 규칙
├── AGENTS.md                      # AI 에이전트 및 기여자 코딩 표준 가이드
├── CLAUDE.md                      # Claude Code 진입점 (AGENTS.md 참조)
├── CONTRIBUTING.md                # 협업 브랜치 전략, 커밋 컨벤션, PR 규칙
├── README.md                      # 프로젝트 개요 및 빠른 시작 가이드
├── pyproject.toml                 # 의존성 패키지 명세, 도구 설정 및 CLI 콘솔 스크립트 정의
├── uv.lock                        # 패키지 버전 고정 락파일
│
├── configs/                       # 파이프라인 전역 실행 및 하이퍼파라미터 YAML 설정
│   └── config.yaml                # 파이프라인 기본 실행 설정 파일
│
├── busan_metro_congestion/        # 핵심 파이썬 소스 패키지
│   ├── __init__.py
│   ├── config.py                  # 프로젝트 전역 경로(PROJECT_ROOT/CONFIGS_DIR), 디바이스 및 시드 설정
│   ├── run_pipeline.py            # E2E 배치 파이프라인 실행 진입점 (CLI 스크립트 매핑)
│   │
│   ├── preprocess/                # 데이터 소스별 전처리 모듈
│   │   ├── __init__.py
│   │   ├── subway.py              # 지하철 승하차 데이터 정제 및 시간대별 통행 압력/혼잡도 산출
│   │   ├── weather.py             # 기상청 단기예보 정제 및 결측 보정
│   │   ├── festival.py            # 부산 축제 정보 1.5km 반경 공간 매핑
│   │   └── features.py            # 데이터 결합, Cyclical/Target 인코딩 및 최종 Parquet 출력
│   │
│   └── modeling/                  # 머신러닝/딥러닝 모델링 및 평가 모듈
│       ├── __init__.py
│       ├── baseline.py            # Fast Track: Random Forest / LightGBM 회귀 모델 학습
│       ├── lstm_net.py            # Accurate Track: PyTorch 시계열 LSTM 네트워크 및 Dataset
│       ├── ensemble.py            # Two-Track 가중치 블렌딩 앙상블 (RF + LGBM + LSTM)
│       └── metrics.py             # 논문 평가 지표 함수 (MAE, RMSE, R², Pearson r, WAPE, Peak MAE)
│
├── tests/                         # 단위 및 통합 테스트 슈트
│
├── artifacts/                     # 모델 산출물 및 평가 결과 저장소
│   ├── baseline/
│   │   ├── checkpoints/           # 학습된 모델 가중치, Scaler 및 체크포인트 (.gitignore 격리)
│   │   └── metrics.json           # 모델 평가 지표 파일 (metrics_{horizon}.json 및 최신 metrics.json)
│   └── ablation/                  # 5대 Feature Group(Set 0~4) 2-Track 롤링·홀드아웃 평가 산출물 및 실행 시각별 결과 표
│       └── checkpoints/           # Ablation 튜닝·트랙별 체크포인트 및 Optuna DB (.gitignore 격리)
│
├── data/                          # 데이터 생명주기 격리 디렉터리
│   ├── reference/                 # 룩업 테이블, 격자 위경도 매핑, 달력 마스터 참조 파일
│   ├── raw/                       # 원본 데이터 소스별 격리 (.gitignore 격리)
│   │   ├── subway/                # 원본 지하철 승하차 데이터
│   │   ├── weather/               # 원본 기상청 단기예보 데이터 및 활용가이드
│   │   ├── festival/              # 원본 축제·전시 데이터 및 활용가이드
│   │   └── calendar/              # 원본 한국천문연구원 특일 정보(공휴일) JSON 및 활용가이드
│   ├── interim/                   # 불필요 데이터 제거 및 스키마 정제 데이터 (.gitignore 격리)
│   │   ├── subway/
│   │   ├── weather/
│   │   └── festival/
│   ├── processed/                 # 유형별 전처리 완료 데이터 (.gitignore 격리)
│   ├── features/                  # 모델 학습용 최종 피처 데이터 (.gitignore 격리)
│   └── samples/                   # 로컬 개발 및 CI 검증용 미니 샘플 (subway, weather, festival, calendar, features)
│
├── docker/                        # GPU 배치 학습 컨테이너 환경
│   ├── Dockerfile                 # NVIDIA CUDA 12 + uv 환경 정의
│   └── docker-compose.yml         # GPU 자원 할당 및 볼륨 마운트 정의
│
├── docs/                          # 프로젝트 문서 및 연구 자료
│   └── DIRECTORY_STRUCTURE.md     # 본 문서 (디렉터리 및 모듈 명세)
│
├── scripts/                       # 데이터 수집, 마스터 빌더 및 2-Track 롤링 평가 CLI 스크립트
└── scratch/                       # 로컬 실험 및 연구용 임시 작업 디렉터리 (.gitignore 격리)
```
