"""
프로젝트 전역 경로 및 실행 환경 설정 모듈
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

# 1. 프로젝트 기준 경로 정의 (상대 경로 기준 자동 산출)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 데이터 디렉터리 체계
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
FEATURES_DATA_DIR = DATA_DIR / "features"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

# 산출물 및 작업 디렉터리
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# 2. PyTorch GPU / CPU 동적 감지 (CUDA 지원 및 CPU 폴백)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_api_key(env_path: Path | None = None) -> str:
    """환경변수 또는 .env 파일에서 PUBLIC_DATA_PORTAL_KEY를 로드합니다."""
    from dotenv import load_dotenv

    target_env = env_path if env_path is not None else PROJECT_ROOT / ".env"
    if target_env.exists():
        load_dotenv(target_env)
    api_key = os.getenv("PUBLIC_DATA_PORTAL_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "PUBLIC_DATA_PORTAL_KEY 환경변수가 설정되어 있지 않습니다. .env 파일을 확인하세요."
        )
    return api_key


def set_seed(seed: int = 42) -> None:
    """재현성을 위한 전역 난수 시드 고정 함수"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
