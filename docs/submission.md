# 모델 제출 자료 준비

이 문서는 Qwen3.5-4B 기반 v2 preference adapter와 재현 자료를 로컬 ZIP으로 묶는 절차입니다. 업로드·배포·공개는 수행하지 않습니다. 패키지는 adapter의 학습 상태와 기존 평가 파일의 실제 상태를 기록하며 제출 가능 또는 사용자 성능이 입증됐다고 표시하지 않습니다.

## 현재 기록

- 공식 기반 모델: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- 기반 파일은 `artifacts/models/` 아래에서만 사용합니다. ZIP에는 모델 가중치가 들어가지 않습니다.
- full training adapter와 학습 로그가 아직 생성되지 않았으면 패키징 dry-run은 누락 상태를 출력하고 종료 코드 2를 반환합니다. Smoke-only adapter는 제출 자료로 거부됩니다.
- `TEAM_INPUTS.md`의 사람 검수 gold set, 실제 조건 표본, 검토 정책, 모델 라이선스 승인과 팀 심사 자료는 아직 팀 제공 대기 항목입니다. synthetic 평가를 사람 평가로 부르지 않습니다.
- 실제 Windows CUDA 학습 120단계와 고정 test 비교를 완료했습니다. 이후 Docker/WSL 복구 후 Linux GPU 서빙도 검증했습니다. [실행 결과](../reports/RESULTS.md)와 [Docker 검증](../reports/DOCKER_VERIFICATION.md)을 구분합니다. 컨테이너 전체 학습을 다시 실행한 것은 아닙니다.

## 기반 모델 다운로드

Python 3.12 환경에서 Hugging Face Hub 클라이언트를 준비한 뒤 실행합니다. 다운로드는 고정된 저장소 ID와 commit SHA만 사용합니다. 기본 Hugging Face 캐시를 이용해 중단된 다운로드를 재개하며 필요한 설정·tokenizer 파일과 index가 참조하는 safetensors shard만 `artifacts/models/qwen3.5-4b-base/`로 복사합니다. 코드·원격 모델 파일을 실행하지 않습니다.

```powershell
python -m pip install huggingface_hub
py -3.12 scripts/prepare_base.py
```

실행 후 `artifacts/models/base-manifest.json`의 파일별 크기와 SHA-256, 공식 LFS SHA가 확인된 파일, 그리고 `base-download.json`의 revision과 다운로드 상태를 확인합니다. 기존 snapshot과 manifest가 있으면 도구는 검증만 하고 덮어쓰지 않습니다. 이 저장소의 현재 base 파일은 이미 확인된 것으로 기록되어 있으므로 재다운로드할 필요가 없습니다.

## 학습과 평가

전체 학습은 `configs/train.json`의 seed·step·precision·데이터 설정을 사용합니다. 이 명령은 GPU 학습을 시작하므로 장치가 준비되고 별도로 실행하기로 한 뒤에만 실행합니다.

```powershell
py -3.12 -m training.train --config configs/train.json
```

`configs/evaluation.json`은 기본적으로 외부 제공자와 Qwen service 연결을 끄고, 기존 test split에서 최대 26개 예제를 사용합니다. 실제 비교 단계별 설정은 evaluation-local/gemini/ensemble.json입니다. 외부 호출이 켜진 두 설정은 예산 안에서 실제 API를 호출하므로 개인 key와 서비스 상태를 먼저 확인합니다.

```powershell
py -3.12 -m evaluation.run --config configs/evaluation.json
```

평가 산출물은 synthetic provenance와 strict semantic tuple 일치 지표를 구분합니다. 이 지표는 사람의 정확도나 출시 준비도를 증명하지 않습니다. 사람이 독립적으로 작성·검수한 test set과 심사 기준은 별도 입력이 필요합니다. 보고서가 없으면 패키지 metadata는 `not_run_or_not_available`을 기록합니다.

## 로컬 설정

키가 필요한 제공자를 사람이 직접 시험하는 경우에만 AI 저장소 루트의 개인 `.env`를 사용합니다. 아래는 필드 이름과 자리표시자 예시입니다. 실제 값은 파일을 채팅·Git·ZIP에 넣지 않습니다.

```dotenv
GEMINI_API_KEY=<enter-your-key-locally>
GEMINI_MODEL=<approved-model-name>
PREFERENCE_AI_PROVIDER=<gemini-or-qwen-or-ensemble>
QWEN_PREFERENCE_URL=http://127.0.0.1:18117/v1/interpret
```

`127.0.0.1`은 로컬 테스트 주소 예시일 뿐 모델 식별자가 아닙니다. Docker 안에서는 같은 Compose network의 service name을 사용할 수 있고, 외부 호스트의 임의 IP를 모델 식별자로 기록하지 않습니다. 기본 평가 설정은 external provider 호출을 비활성화합니다. BE의 provider/URL 설정은 BE `.env`에, native Qwen/evaluation CLI 설정은 AI `.env`에 넣습니다. 프로세스 환경변수가 우선합니다.

## 제출 패키지 생성

학습과 provenance 검사 결과를 먼저 살펴봅니다. 누락된 adapter나 manifest, smoke-only 상태는 dry-run 오류에 구체적으로 표시됩니다.

```powershell
py -3.12 scripts/package_submission.py --dry-run
```

검증을 통과하면 ZIP과 sidecar SHA-256 파일을 AI 저장소 `artifacts/submission/`에 생성합니다.

```powershell
py -3.12 scripts/package_submission.py
```

ZIP에는 adapter와 manifest, training log/config, dataset과 manifest, official base ID/revision 및 license notice, 기존 evaluation report, `TEAM_INPUTS.md`, FE·BE·AI commit SHA, 재현 안내, 그리고 ZIP 안 파일들의 상대 경로·크기·SHA-256 manifest가 포함됩니다. ZIP 자체 checksum은 `.sha256` sidecar에 기록합니다. adapter의 로컬 base 경로는 공식 모델 ID로 정리하고 report/log의 저장소 절대 경로는 상대 경로로 바꿉니다.

각 저장소를 기본 이름으로 나란히 clone하면 configs/submission.json의 경로를 사용할 수 있습니다. worktree 이름이 다르면 이 설정을 로컬 artifacts/ 아래에 복사해 source_repositories만 실제 체크아웃으로 바꾸고 `--config`로 지정합니다. 실제 제출할 코드의 commit을 사용하며 원래 main의 구버전을 잘못 기록하지 않습니다.

JSON과 CSV가 일치하지 않는 평가 파일, 형식/출처가 불명확한 파일, 재배포 허가가 없는 사람 평가 파일은 제외하고 이유를 기록합니다. 사람 원문·검수 metadata·정답을 ZIP에 넣는 허가로 합성 데이터 사용 허가를 대신하지 않습니다.

ZIP에는 base weights, `.env`, secret, 다운로드·빌드 cache, `node_modules`, venv, 원본 절대 경로, 사용자 폴더 이름이 들어가지 않습니다. 경로를 안전하게 정리할 수 없는 텍스트 artifact는 패키징을 중단합니다. 소스 코드는 ZIP에 복사하지 않으며 FE·BE·AI commit SHA와 재현 문서를 함께 기록합니다.

패키징 도구는 모델을 재평가하거나 사람 승인 상태를 채우지 않습니다. 내용·license·평가·팀 심사 입력을 검토한 뒤 별도 승인 절차를 진행해야 합니다.
