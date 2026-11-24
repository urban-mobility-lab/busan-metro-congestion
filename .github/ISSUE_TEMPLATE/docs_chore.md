---
name: 문서·인프라·데이터 갱신 (docs / chore)
about: 문서 정비, CI/Docker 설정, 참조 데이터 갱신 작업을 추적합니다.
title: "[chore] "
labels: "track:main"
assignees: ""
---

## 작업 개요
- 변경 대상 문서, 인프라 설정 또는 참조 데이터 갱신 내용을 작성합니다.

## 체크리스트
- [ ] 대상 파일 반영
- [ ] 기존 문서/코드/GitHub 설정과 정합성 확인
- [ ] (코드·설정 변경 시) `uv run ruff` 및 `uv run pytest tests/ -v` 통과
