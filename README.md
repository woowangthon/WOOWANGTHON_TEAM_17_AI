# 충북 여행 AI 파서

자유 입력 여행 메모와 고정 여행 조건을 받아 장소 검색 전 확인용 미리보기를 만드는 Python 패키지입니다. Gemini 응답은 `pending_review` 상태로 반환하며, 사용자 확인과 장소 검색·점수화는 이 패키지의 범위에 포함되지 않습니다.

## Gemini + Qwen 취향 분석 v2

`interpret_personal_draft_v2`는 같은 원문을 Gemini와 로컬 Qwen에 독립적으로 전달하고, 조건의 부정·강도·횟수·하루/여행 전체 범위·원문 근거를 보존합니다. 정상 응답 두 개의 의미가 일치한 조건만 자동 적용하고 해석 차이는 `unresolved`로 남깁니다. 모델 간 일치는 정확도 확률이나 사람 검수가 아닙니다. 기존 v1 함수는 유지합니다.

```python
from chungbuk_trip_ai import interpret_personal_draft_v2

result = interpret_personal_draft_v2(original_text, conditions, provider="ensemble")
```

BE는 기존 개인 취향 API의 `X-Preference-Schema: 2`로 사용합니다. 원문·조건 버전, 멱등성 영수증, 수동 수정과 개인 정보 공개 범위는 BE가 검증합니다. FE에서는 같은 입력 화면에서 조건을 수정·저장·직접 확정합니다. 걷기 제한이 없으면 AI가 숫자를 추측하거나 필수 추가 질문을 만들지 않습니다.

- 규격: [원자 조건 v2](docs/atomic-contract.md)
- 데이터·학습·평가: [재현 절차](docs/reproducibility.md)
- 모델 다운로드·제출 ZIP: [제출 자료 준비](docs/submission.md)
- 사람이 나중에 채울 원문·정답: [TEAM_INPUTS.md](TEAM_INPUTS.md)
- 브라우저 검수 범위: [화면 검수 기록](reports/ui/README.md)
- 실제 학습·비교·실패: [실행 결과](reports/RESULTS.md)
- Docker GPU 서빙 확인: [컨테이너 검증](reports/DOCKER_VERIFICATION.md)
- 추후 PPT에 사용할 근거: [발표 메모](reports/PRESENTATION_NOTES.md)

단체 취향의 필수 조건·개인별 제한·동등한 1인 기여·충돌 후보는 BE의 비공개 `groupConditions` 인수 자료에 보존합니다. 새 원자 조건의 장소 선택·날짜 배분·경로 최적화 적용은 일정 알고리즘 담당자의 후속 연동 범위입니다. 이 변경은 TOPTW·2-opt·relocate를 구현하거나 그 제약 준수를 보장하지 않습니다.

공식 Qwen3.5-4B 고정 revision과 별도 LoRA adapter를 사용합니다. 기반 가중치는 Git에 넣지 않습니다. IP나 Docker 내부 주소 대신 코드 커밋·모델 revision·adapter SHA-256·설정과 의존성으로 환경을 재현합니다. 실제 모델·학습·평가 기록과 Docker/WSL 미검증 항목은 제출 보고서에 구분해서 남깁니다.

## 설치

Python 3.12 이상을 사용합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

AI 저장소 루트의 `.env`에 Google AI Studio 키를 입력합니다. `GEMINI_MODEL` 기본값은 `gemini-3.5-flash-lite`이며, 프로세스 환경변수의 `GEMINI_API_KEY`와 `GEMINI_MODEL`이 AI 저장소 `.env` 값보다 우선합니다. 이 패키지는 `.env` 전체를 프로세스 환경에 주입하지 않습니다.

## CLI

JSON 파일을 지정하거나 stdin으로 JSON을 전달할 수 있습니다.

```powershell
chungbuk-trip-preview .\examples\trip_input.json
Get-Content .\examples\trip_input.json -Raw | chungbuk-trip-preview
```

입력에는 `dates.start`, `dates.end`, `transport`, `total_budget_won`, `departure.time`, `departure.region`, `headcount`, `preferences_text`가 필요합니다. `headcount`는 모임방에서 확정된 인원입니다. 출력은 추출 결과와 고정 조건을 포함한 미확정 미리보기입니다.

## 백엔드 연결

백엔드 개발환경에서는 이 패키지를 editable dependency로 설치합니다.

```powershell
python -m pip install -e ..\WOOWANGTHON_TEAM_17_AI
```

백엔드 코드는 다음과 같이 `chungbuk_trip_ai.create_trip_preview(data)`를 호출할 수 있습니다. 배포 환경은 패키지를 설치하고 Gemini 키를 프로세스 환경변수로 주입해야 합니다. 로컬 개발에서는 설정이 AI 저장소 루트 `.env`에서만 읽히므로, 백엔드 작업 디렉터리의 `.env`는 검색되지 않습니다. 설치된 패키지를 사용할 때는 AI 저장소 작업 트리의 editable 설치가 그 저장소 루트를 기준으로 `.env`를 찾습니다.

```python
from chungbuk_trip_ai import create_trip_preview

preview = create_trip_preview(trip_input)
```

확정된 구성원 선호를 신뢰된 지원 태그 목록에 대응시키는 제안은 다음 공개 함수를 사용합니다.
입력은 로컬 별칭(`p1` 등)과 확정된 선호 여섯 필드만 포함하며, 반환된 태그 제안은 모두
`requiresReview=true`입니다. 이 함수는 저장·확정하지 않고, 태그 목록이 비었거나 분류할 텍스트가
없으면 네트워크를 호출하지 않습니다.

패키지 0.3.0의 이 함수는 `walking`을 기존 계약의 문자열 `"20"`, `"40"`, `"unlimited"` 또는 `null`로 받습니다. 분 제한은 장소 사이 개별 도보 구간 기준이며 출력 `maxMinutes`만 정수입니다. 지원 태그는 호출자가 검증한 최대 200개·각 80자 목록입니다. 실제 회원 ID·닉네임·`originalText`는 입력에 허용하지 않습니다.

반환값은 `terms`, `unmapped`, `walking`, `conflictCandidates`, `needsReview`입니다. 중요도는 확정 입력에서 복사하고, 동일 태그의 선호·회피 대립을 검토용 충돌 후보로 묶습니다. 모델이 전체 문장을 근거로 인용해도 모든 의미가 태그에 반영됐다는 보장은 없으므로 각 확정 문장을 `unmapped`에 보존합니다. 제안이 있는 필드는 `coverage_unverified`, 제안이 없는 필드는 `uncovered`입니다. 개인 원문을 다시 해석하거나 자동으로 필터·점수·일정을 실행하지 않습니다.

새 함수의 요청 크기는 최대 400,000바이트, 응답은 기존과 동일하게 256,000바이트, provider 요청 timeout은 60초입니다. 전체 deadline과 처리 중 버전 변경 검사는 호출 측 BE가 담당합니다.

```python
from chungbuk_trip_ai import normalize_confirmed_preferences

normalization = normalize_confirmed_preferences(confirmed_preferences, supported_tags)
```

개인 취향 원문을 해석하려면 다음 공개 함수를 사용합니다. `conditions`는 호출 측에서 권한과 버전을 확인한 뒤 준비 계약의 Conditions JSON을 그대로 전달합니다. 함수는 DB 저장이나 확정을 하지 않습니다. `GeminiInvalidResponseError`, `GeminiUnavailableError`, `GeminiTimeoutError`는 모두 `GeminiError`의 하위 유형이며 BE adapter에서 각각 AI 응답 오류, 이용 불가, 시간 초과로 매핑할 수 있습니다.

```python
from chungbuk_trip_ai import interpret_personal_draft

suggestion = interpret_personal_draft(original_text, conditions)
```

반환값은 `{"suggestions":{"likes":...,"avoid":...,"avoidImportance":...,"food":...,"foodImportance":...,"walking":...,"scheduleItems":[]},"needsWalkingAnswer":...}`입니다. 원문에서 걷기 시간과 걷기/도보 문맥의 명시 근거가 확인되지 않는 `walking` 제안은 `null`과 `needsWalkingAnswer=true`로 낮춥니다. 이 함수는 저장·준비 완료를 수행하지 않으며 실제 저장과 확인은 호출 측 BE/FE가 담당합니다.

동일한 최초 분석 요청에서 실제 예약한 숙소와 고정 약속을 `scheduleItems`로 추출합니다. 각 항목은 `{id,kind:'lodging'|'appointment',scope:'personal'|'group'|null,placeQuery,contentid:null,date,time,endDate,endTime}`입니다. ID는 서버가 생성하며, DB 장소 연결과 본인 입력 보완은 BE가 처리합니다. 숙소 추천 요청은 예약으로 만들지 않습니다. 원문에서 확인되지 않은 날짜·오전/오후·범위는 null이며, 상대 날짜는 주어진 여행 날짜에만 연결합니다. 모델의 근거 문구는 내부 검증에만 쓰고 반환값에서 제거합니다. 체크인/체크아웃 시간이나 좌표를 추측하지 않습니다. 개인 일정이 그룹 일정으로 반영되는 정책은 BE에서 본인/방장 권한과 함께 검증합니다.

새 일정 생성 흐름은 이 최초 분석 결과를 저장·준비 완료한 뒤 사용합니다. 아래 `normalize_confirmed_preferences`는 기존 호출자 호환을 위해 유지하며 새 준비 흐름의 두 번째 필수 AI 단계로 쓰지 않습니다. 계약 원본은 [여행 준비](https://app.notion.com/p/412e977e49088197b345d73603c6a27f)와 [일정 생성·공동 캔버스](https://app.notion.com/p/413e977e4908817a9af9ca9c7fa0ddee)입니다.

## 후보 일정 초안

`draft_candidate_itinerary(dates, candidates, fixed_visits=None, max_visits_per_day=8, schedule_context=None)`는 BE가 검증해 전달한 후보 장소만으로 일정 초안을 요청합니다. 후보 입력에는 `contentid`, `title`, `planningRole`, `adminCity`, `allowedDates`, `groupScore`가 필요하고 `coordinate`와 `estimatedStayMinutes`는 선택입니다. schedule_context에는 날짜별 시작·마감 시각, 출발·복귀 지점, 확인된 고정 일정의 시각을 전달할 수 있습니다. 모델이 식당을 빠뜨리면 후보 중 해당 날짜에 방문할 수 있는 식당을 보완합니다. 실제 경로 시간 검증은 BE가 담당합니다. `fixed_visits`는 날짜, 장소 ID, 0부터 시작하는 위치를 지정합니다.

응답은 날짜별 `{date, contentids}`만 허용하며 모든 날짜, 후보 허용일, 중복 ID, 고정 방문 위치를 검증합니다. 결과는 항상 `requiresReview=true`, `routeVerified=false`, `timeVerified=false`, `budgetVerified=false`입니다. 이 함수는 경로·시간·예산을 계산하거나 결과를 저장하지 않습니다. `validate_candidate_itinerary(...)`는 동일한 검증을 provider 없이 적용합니다.

```python
from chungbuk_trip_ai import draft_candidate_itinerary

draft = draft_candidate_itinerary(dates, candidates, fixed_visits)
```

## 요청 크기와 네트워크

Gemini API 호출은 `generativelanguage.googleapis.com`으로 HTTPS 요청을 보내며 API 키는 헤더로 전달합니다. 기존 여행 미리보기 입력 요청은 24,000바이트, 개인 취향 해석은 원문/Conditions 길이를 수용하도록 64,000바이트, 일정 초안 요청은 최대 400,000바이트, 모든 읽기 응답은 256,000바이트로 제한합니다. 일정 초안 provider timeout은 60초, 최대 출력은 16,384 tokens이며 자동 재시도하지 않습니다. API 오류 본문이나 키는 사용자 표시 오류에 포함하지 않습니다. 실제 API 호출 없이 테스트하려면 저장소 루트에서 패키지 테스트를 실행합니다.
