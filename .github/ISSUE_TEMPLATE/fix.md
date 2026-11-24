---
name: 버그 수정 (fix)
about: 코드 오류, 테스트 실패, 데이터 정합성 문제를 보고하거나 수정합니다.
title: "[fix] "
labels: "track:main"
assignees: ""
---

## 문제 현상
- 발생한 오류 메시지, 증상 또는 비정상 동작을 설명합니다.

## 재현 방법
1. 실행 명령어 또는 입력 조건
2. 실제 결과 vs 기대 결과

## 원인 분석 및 수정 계획
- [ ] 원인 파악:
- [ ] 수정 및 회귀 테스트 추가(`tests/`):

## 완료 조건
- [ ] `uv run ruff format --check .` 및 `uv run ruff check .` 통과
- [ ] `uv run pytest tests/ -v` 통과
- [ ] PR 본문에 `Closes #<이슈번호>` 연결 및 CI 통과
