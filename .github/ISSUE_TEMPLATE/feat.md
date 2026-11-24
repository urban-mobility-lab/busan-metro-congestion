---
name: 기능 및 파이프라인 개발 (feat)
about: 새로운 데이터 수집, 전처리, 모델링, 파이프라인 기능을 제안하거나 추적합니다.
title: "[feat] "
labels: "track:main"
assignees: ""
---

## 작업 목표
- 무엇을 구현하고 왜 필요한지 간결하게 작성합니다.

## 분류 (라벨 지정)
- 트랙: `track:main` (운영 배치) / `track:research` (논문 실험·튜닝)
- 영역: `area:preprocess` / `area:modeling` / `area:data` / `area:api` / `area:ui` / `area:ops` / `area:ci` / `area:docs`

## 구현 범위 및 변경 예정 파일
- [ ] 주요 모듈/스크립트:
- [ ] 테스트 코드(`tests/`):
- [ ] 샘플/참조 데이터(`data/samples/`, `data/reference/`):

## 완료 조건 (Definition of Done)
- [ ] `uv run ruff format --check .` 및 `uv run ruff check .` 통과
- [ ] `uv run pytest tests/ -v` 통과 (`data/samples/`만 사용)
- [ ] PR 본문에 `Closes #<이슈번호>` 연결 및 CI 통과
