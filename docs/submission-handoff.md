# Qwen 취향 분석 adapter 인수 자료

이 ZIP은 학습된 연구용 adapter와 재현 근거다. 기반 모델 전체, API key, 사람 정답은 포함하지 않는다. 공개 출시·사람 검수 승인을 뜻하지 않는다.

1. [코드 commit](provenance/source-commits.json)에 기록된 FE·BE·AI 저장소 버전을 준비한다. AI 학습 당시 source commit은 adapter manifest에 별도로 기록돼 있다.
2. ZIP 자체의 `.sha256` sidecar와 [내부 파일 manifest](provenance/submission-files.json)의 크기·SHA-256을 대조한다. manifest 자신은 내부 목록에서 제외한다.
3. AI 저장소에서 Python 3.12 venv를 만들고 CUDA 12.8용 Torch 2.9.1 및 [직접 의존성](reproduction/requirements-training.txt)을 설치한다. [Windows 실측 lock](reproduction/requirements-lock.windows.txt)도 제공한다.
4. AI 저장소 루트에서 `python -m scripts.prepare_base`를 실행한다. 기반 모델은 Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`다. 공식 파일 hash와 라이선스를 확인한다.
5. ZIP을 AI 저장소 `artifacts/submission/unpacked/` 같은 별도 디렉터리에 푼다. 기존 학습 adapter를 덮어쓰지 않고, AI `.env`의 `QWEN_ADAPTER_PATH=artifacts/submission/unpacked/adapter`로 해당 adapter를 지정한다.
6. `python -m serving.server`로 실행하고 `http://127.0.0.1:18117/ready`의 ready/adapterLoaded를 확인한다. BE 환경에는 v2 AI 패키지, 개인 Gemini key, `PREFERENCE_AI_PROVIDER=ensemble`, `QWEN_PREFERENCE_URL=http://127.0.0.1:18117/v1/interpret`가 필요하다.

학습을 다시 실행할 필요 없이 adapter를 서빙할 수 있다. 다시 학습하려면 같은 데이터·설정·seed를 사용하되 새 output 디렉터리로 분리한다. 다른 OS·드라이버·커널에서 bitwise 동일한 결과는 보장하지 않는다. WSL 복구 후 Docker 이미지 빌드·GPU 추론을 확인했으며 [Linux lock](reproduction/requirements-lock.linux.txt)과 [검증 기록](evidence/DOCKER_VERIFICATION.md)을 제공한다. 컨테이너 전체 학습을 재실행한 것은 아니다.

## 근거 파일

- [실험 결과와 한계](evidence/RESULTS.md), [수치·원자료 SHA](evidence/EVALUATION_SUMMARY.json)
- [발표 메모](evidence/PRESENTATION_NOTES.md), [실제 검사 범위](evidence/VALIDATION.md)
- [평가 원자료 JSON/CSV](evaluation/reports/), [학습 설정](training/config.json), [학습 로그](training/training.jsonl)
- [원본 adapter manifest](training/adapter-manifest.json), [패키지 adapter manifest](adapter/adapter-manifest.json)
- [공식 모델 라이선스](notices/Qwen-Qwen3.5-4B/LICENSE), [추후 사람이 채울 자료](team/TEAM_INPUTS.md)

일부 상세 문서의 상대 경로는 Git 체크아웃 기준이다. 전체 코드·원래 문서는 위 source commit에서 확인한다. 사람 자료·운영 인증/MySQL·실제 모델 브라우저 전체 인수·최종 웹 배포는 별도다. 합성 시험의 정밀도 80%를 사용자 정확도 80%라고 표현하지 않는다.
