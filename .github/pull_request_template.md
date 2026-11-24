## 관련 이슈
Closes #

## 변경 요약
- 무엇을, 왜 변경했는지 핵심 사항을 2~4줄로 요약합니다.

## 작업 트랙 및 영역
- [ ] `track:main` (운영 배치 경로) / [ ] `track:research` (연구·실험 경로)
- [ ] `area:preprocess` / [ ] `area:modeling` / [ ] `area:data` / [ ] `area:ci` / [ ] `area:docs`

## 로컬 검증 체크리스트 (PR 제출 전 필수)
- [ ] `uv run ruff format --check .` 통과
- [ ] `uv run ruff check .` 통과
- [ ] `uv run pytest tests/ -v` 통과 (`data/samples/` 격리 테스트)
- [ ] `git add -A` 없이 해당 PR 파일만 명시적으로 스테이징 확인
- [ ] 대용량 데이터(`data/raw/**`, `data/features/**`), 가중치(`checkpoints/**`), `.env` 미포함 확인
