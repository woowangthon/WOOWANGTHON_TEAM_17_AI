# 고도화 최종 검증 기록

2026-11-14 로컬 체크아웃 기준이다. 운영 배포와 PR 병합을 뜻하지 않는다.

| 범위 | 실제 실행 | 결과 |
|---|---|---|
| AI | Python 3.12 venv, `python -m pytest -q` | 106 passed |
| AI | `python -m compileall -q chungbuk_trip_ai training serving evaluation scripts` | 통과 |
| Docker 문법 | `docker compose --profile train --profile serve config --quiet` | 통과; engine·이미지 실행 검사 아님 |
| BE | 기존 BE venv에 AI 작업 경로를 지정, `pytest -m "not mysql" -q` | 229 passed, 177 deselected; 기존 deprecation warning |
| FE | `npm test`, `npm run check` | 227개 검사, 타입·lint·빌드 통과; 기존 lint warning |
| UI | 1440×1000 / 768×1024 / 390×844 | 합성 SQLite·모의 모델로 분석·직접 수정·저장·확정·복원, 키보드·오류·넘침 확인 |
| GPU | 실제 Qwen 4B LoRA 120단계 및 adapter 재로드 | 완료; source/data/config/weight hash 기록 |
| 모델 평가 | 같은 고정 합성 test 26개, 5개 방식 | 실행 완료; v1은 atomic F1 비교 제외 |
| BE 실제 모델 smoke | SQLite/test-auth + 실제 Gemini/Qwen | 공통 fixture 텍스트를 지운 focused 조건에서 6개 HTTP 단계 통과; 원래 fixture strict 실패도 보존 |
| 독립 검토 | Sol high, 구현 worker와 별도 | 실제 SDK hash, source hash, schema/receipt, CPU loss, HTTP deadline, 사람 자료 권한, 보고서 CSV/JSON 일치·경로 처리 지적 수정·재검토 |

모델 호출·GPU 학습과 오프라인 단위 검사를 혼합해 같은 테스트 수로 세지 않는다. 사람 정답·사람 인수 검수는 0개/미실행이다. 새로운 방문 횟수/날짜 제약의 경로 적용은 별도 일정 담당 범위다.

Docker Desktop의 WSL 목록 조회는 12초 제한을 넘겼고 engine pipe가 없었다. 데이터 초기화·WSL 배포판 삭제·운영 DB 변경 없이 중단했다. Docker 이미지 빌드/GPU 실행, 전용 MySQL 177개, 운영 인증/실제 모델 브라우저 전체 인수는 미검증이다.

원래 세 저장소와 run.py는 전환하지 않았다. 작업은 별도 ensemble-AI/BE/FE worktree와 기능 브랜치에 있다. 원래 FE의 미커밋 캔버스 변경도 유지했다. 공개 push·Draft PR·adapter 업로드는 사용자의 제목·본문 사전 검토 절차가 끝난 뒤 진행한다.
