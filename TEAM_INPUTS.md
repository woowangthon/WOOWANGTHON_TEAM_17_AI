# 사람이 나중에 채울 자료

현재 205개 원문·정답은 재현 가능한 합성 데이터다. 사람이 작성·검수한 정답은 **0개**이며, 실제 사용자 정확도나 전문가 검증 자료로 표시하지 않는다.

## 독립 평가용 입력과 정답

`data/human/annotation-template.json`을 `data/human/inbox.local.json`으로 복사해 작성한다. 이 로컬 파일과 변환한 JSONL은 Git에서 제외한다.

1. 학습 예시를 바꿔 쓰지 말고 새 여행 요청을 직접 작성한다. 가능하면 팀원이 각각 작성한 원문을 모은다.
2. 입력 당시의 공통 여행 조건을 `conditions`에 적고, 원문에서 확인되는 조건만 `target`에 표시한다. 해석할 수 없으면 `unresolved`에 남긴다.
3. `reviewMetadata.annotatorId`와 `reviewMetadata.reviewerId`에 서로 다른 `anon-...` 익명 코드를 적고, 작성자와 별도 검수자가 숫자·부정·필수/선호·하루/여행 전체·개인/단체 범위를 확인한다.
4. 실제 `sourceOwner`, `sourceLicense`, 시간대가 포함된 `reviewedAt`을 적는다. `attestations`는 사람이 새 원문을 쓰고 독립적으로 검수했을 때만 true로 둔다. `permissions`의 local evaluation 허가를 확인하고 외부 AI 처리·공개 재배포는 받은 허가만 true로 설정한다. 빈 값이나 `pending`은 검수 완료가 아니다.

설명은 [사람 평가 자료 안내](data/human/README.md)와 [원자 조건 규격](docs/atomic-contract.md)에 있다. 작성 후 `python scripts/import_human.py --input data/human/inbox.local.json --output data/human/human-evaluation-v2.jsonl`로 test-only 자료를 검증·봉인한다. 생성 manifest는 현재 합성 데이터 SHA-256을 기록하며, 사람 평가 JSONL은 학습 데이터로 합치지 않는다. 작성 전까지 인간 평가 수치는 빈 값으로 유지한다. 평가용 원문은 학습 파일에 합치지 않는다.

## 실제 사용 검수 기록

팀원이 실제 서비스에서 입력·분석·수정·저장·확정을 수행한 날짜와 발견한 문제를 `reports/HUMAN_ACCEPTANCE.md`에 적는다. 자동 검사·합성 여행방 브라우저 검수와 구분한다.

## Docker/WSL 후속 상태

초기 Docker Desktop의 `Unexpected WSL error`와 WSL 명령 시간 초과 이후 Windows/WSL이 정상화됐다. 후속으로 이미지 빌드·GPU cold-JIT·기존 adapter의 실제 HTTP 추론을 확인했다. Docker 데이터를 초기화하거나 배포 DB를 변경하지 않았다. 컨테이너 전체 학습과 전용 MySQL 검수는 아직 실행하지 않았다. 근거는 [Docker 검증](reports/DOCKER_VERIFICATION.md)에 있다.

## 제출자가 최종 기록할 값

- 공개 FE·BE·AI 저장소와 제출할 main 커밋 해시
- 실제 웹 URL, 발표 PDF·영상 링크, 팀 Notion 접근 권한
- 공개 다운로드 가능한 학습 adapter 파일과 SHA-256

IP나 Docker 내부 IP는 모델의 재현 정보가 아니다. 모델 ID·고정 revision·adapter 해시·코드·설정·필요 GPU 조건을 함께 제공한다. API 키는 개인 `.env`에만 넣는다.
