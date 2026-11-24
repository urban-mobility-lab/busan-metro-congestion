"""
전역 설정(config.py) 프로젝트 루트, 디바이스 및 시드 고정 단위 테스트
"""

import random

import numpy as np
import torch

from busan_metro_congestion.config import (
    DEVICE,
    PROJECT_ROOT,
    set_seed,
)


def test_config_project_root_resolves_repository_root():
    """PROJECT_ROOT가 실제 저장소 루트 디렉터리를 가리키는지 검증."""
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "busan_metro_congestion" / "config.py").is_file()


def test_config_device_and_seed_reproducibility():
    """실행 디바이스 감지 분기 일치 및 set_seed() 난수 재현성(random/numpy/torch) 검증."""
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert isinstance(DEVICE, torch.device)
    assert DEVICE.type == expected_device

    set_seed(123)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(5)
    torch_a = torch.rand(5)

    set_seed(123)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(5)
    torch_b = torch.rand(5)

    assert py_a == py_b
    assert np.array_equal(np_a, np_b)
    assert torch.equal(torch_a, torch_b)
