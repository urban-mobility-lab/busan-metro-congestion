# CUDA GPU 가속 배치 학습 컨테이너 환경
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# uv 바이너리 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 의존성 파일 먼저 복사하여 Docker 레이어 캐싱 극대화
COPY pyproject.toml uv.lock ./

# uv를 사용하여 결정론적 의존성 설치 (lock 파일 기반, 개발 의존성 제외, 프로젝트 자체 제외)
RUN uv sync --frozen --no-dev --no-install-project

# 전체 프로젝트 코드 복사
COPY . .

# 패키지 및 CLI 엔트리포인트(busan-metro) 설치
RUN uv sync --frozen --no-dev

# 기본 실행 명령 (E2E 배치 파이프라인 CLI)
CMD ["busan-metro"]
