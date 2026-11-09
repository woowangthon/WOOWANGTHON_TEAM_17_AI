# 충북 여행 AI 파서

자유 입력 여행 메모와 고정 여행 조건을 받아 장소 검색 전 확인용 미리보기를 만드는 Python 패키지입니다. Gemini 응답은 `pending_review` 상태로 반환하며, 사용자 확인과 장소 검색·점수화는 이 패키지의 범위에 포함되지 않습니다.

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

반환값은 `{"suggestions":{"likes":...,"avoid":...,"avoidImportance":...,"food":...,"foodImportance":...,"walking":...},"needsWalkingAnswer":...}`입니다. 원문에서 걷기 시간과 걷기/도보 문맥의 명시 근거가 확인되지 않는 `walking` 제안은 `null`과 `needsWalkingAnswer=true`로 낮춥니다. 제안에는 `originalText`가 포함되지 않으며 사용자가 검토하기 전에 저장하지 않습니다.

## 요청 크기와 네트워크

Gemini API 호출은 `generativelanguage.googleapis.com`으로 HTTPS 요청을 보내며 API 키는 헤더로 전달합니다. 기존 여행 미리보기 입력 요청은 24,000바이트, 개인 취향 해석은 원문/Conditions 길이를 수용하도록 64,000바이트, 읽는 응답은 256,000바이트로 제한합니다. API 오류 본문이나 키는 사용자 표시 오류에 포함하지 않습니다. 실제 API 호출 없이 테스트하려면 저장소 루트에서 패키지 테스트를 실행합니다.
