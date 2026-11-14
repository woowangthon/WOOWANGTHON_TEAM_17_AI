# Docker GPU 서빙 검증 — 2026-11-14

Docker/WSL 복구 후 학습한 Qwen adapter를 실제 Linux GPU 컨테이너에서 실행했다. 모델을 재학습하거나 Gemini를 호출한 검사가 아니다.

## 실행 상태와 재현

- Docker Desktop 4.99.0, Engine 29.9.3, WSL 2 Linux backend.
- 프로젝트 `ensemble-ai`, 서비스 `preference-qwen`, 실행 컨테이너 `ensemble-ai-preference-qwen-1`.
- 검증 시 상태 `running / healthy`, GPU RTX 4080 SUPER, `cuda:0`.
- 이미지 ID `sha256:95e3b4e7eb9e336a42c3aeec18bec109974857e407752554a24b34431f3e9d94`.
- 실행 코드의 기능 commit `a91afc726cf4478a4d7cc35779d611c2567a20d6`.
- 기반 모델 revision과 기존 학습 adapter manifest를 검증한 뒤 로드했다. 가중치는 새로 학습하지 않았다.

AI 저장소 루트에서 실행한다.

```powershell
docker compose --profile serve build preference-qwen
docker compose --profile serve up -d preference-qwen
docker compose --profile serve ps
docker compose --profile serve logs --tail 30 preference-qwen
```

Docker Desktop의 Containers에서 프로젝트를 펼쳐 해당 컨테이너의 상태·로그를 확인한다. `http://127.0.0.1:18117/ready`에서 ready=true, mode=adapter, adapterLoaded=true를 확인할 수 있다. GPU 컨테이너의 Windows WSL 2 요구 사항은 [Docker 공식 안내](https://docs.docker.com/desktop/features/gpu/)를 따른다.

## 실제 검사

| 검사 | 결과 |
|---|---|
| 이미지 빌드 | 통과 |
| Linux 의존성 | lock의 63개 package version과 실행 컨테이너 일치, pip check 통과 |
| CUDA·driver | GPU 사용 가능, libcuda.so.1 로드 성공 |
| 새 Triton cache | gcc·Python.h 존재, kernel을 새로 컴파일하고 GPU 계산 결과 일치 |
| 모델 준비 | ready=true, adapterLoaded=true, 기존 adapter manifest SHA 일치 |
| 실제 HTTP 추론 | 개발용 초밥 1회 입력에 HTTP 200; 조건 1개, unresolved 0개 |
| 첫 요청 / 동일 입력 재요청 | 약 30.99초 / 8.73초; 각 1회 관측이며 성능 통계가 아님 |
| 격리 | artifacts는 read-only mount, 공개 port는 127.0.0.1:18117 |
| 이미지 내용 | 개인 .env·Git·artifact 파일 없음; 가중치는 host mount로 제공 |
| 제공자 설정 | Qwen 컨테이너에 Gemini key 없음; HF/Transformers offline 설정 |

근거: [초기 실패](live/docker-runtime-initial.json), [보완 후 cold-JIT](live/docker-runtime-verified.json), [의존성·이미지 점검](live/docker-environment-check.json), [실제 추론](live/docker-inference-smoke.json).

최초 빌드에는 gcc·Python 헤더와 python-dotenv가 빠져 있었다. 필요한 apt package를 추가하고, Linux lock과 함께 프로젝트 의존성을 설치한 뒤 pip check를 빌드 단계에 넣었다. 큰 CUDA/PyTorch 의존성 layer는 재사용했다. 컴파일러 요구는 [Triton 3.5.1 build 코드](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/runtime/build.py)와 실제 cold-JIT 실행으로 대조했다.

## 범위

Linux GPU 추론 운영 검수다. 전체 120단계 학습을 컨테이너에서 재실행하지 않았고, 기존 Windows 합성 품질 평가를 Linux 품질 평가로 바꾸지 않는다. 운영 MySQL·실제 인증·FE/BE/Gemini까지 연결한 브라우저 전체 인수와 사람 정답 평가는 여전히 별도다. adapter는 trained-not-reviewed 상태를 유지한다.
