# Medical Lecture Processor

의학 강의 오디오를 전사하고, 족보 `급분바`와 과목 prior를 이용해 STT 오인식을 보정하는 파이프라인입니다.

현재 이 프로젝트는 "강의노트 자동 생성기"보다 "의학 강의 전사 교정기"로 보는 편이 맞습니다.

## 운영 원칙

- 목표는 `환각 없이`, `원문에 가깝게`, `의학용어만 보정`하는 것
- 기본 운영은 `Whisper -> corrected transcript`
- 포맷 레이어는 기본 비활성화
- 같은 transcript를 여러 번 튜닝할 수 있게 `reprocess` UX를 우선
- `medical map`은 완전한 RAG가 아니라, 가벼운 힌트 주입 레이어

## 현재 추천 플로우

가장 추천하는 기본 실행:

```bash
python main.py --correction-only --openrouter --compress-for-api
```

이 모드는 다음 순서로 동작합니다.

1. OpenAI Whisper API로 전사
2. 족보 PDF에서 `급분바` 섹션 추출
3. 강의명과 족보를 바탕으로 `medical map` 생성
4. OpenRouter LLM으로 STT 오인식만 교정
5. `*_corrected_transcript.txt` 저장

기본적으로는 여기까지만 쓰는 것이 가장 실용적입니다.

## 왜 이렇게 쓰는가

현재 경험상 병목은 점점 `LLM 포맷팅`보다 `원음 품질`과 `STT` 쪽에 있습니다.

- 환각 억제는 이미 충분히 잘 되는 편
- 의학용어 복원도 correction layer에서 꽤 잘 되는 편
- lecture notes 포맷 레이어는 비용 대비 체감 효용이 낮은 경우가 많음
- 따라서 기본은 `raw -> corrected transcript`까지만 생성하는 것이 낫습니다

즉 지금은 "예쁘게 정리된 노트"보다 "믿을 수 있는 교정본"이 우선입니다.

## 비용 감각

현재 기본 운영인 `Whisper API + correction-only + subject router` 기준 대략 비용은 이 정도로 보면 됩니다.

- 50분 강의 1개: 약 `$0.33 ~ $0.35`
- 대부분은 Whisper 전사 비용
- correction layer는 보통 몇 센트 수준
- 제목 subject router 비용은 사실상 무시 가능

대략적인 구성:

- Whisper API: 50분 기준 약 `$0.30`
- correction layer: 보통 약 `$0.03 ~ $0.04`
- subject router: 거의 `0`

즉 비용을 줄이려면 가장 중요한 건:

1. Whisper를 다시 안 태우고 `--reprocess`를 쓰는 것
2. 포맷 레이어를 기본적으로 끄는 것

참고로 이 수치는 2026년 3월 기준 가격과 최근 실제 실행 로그 평균을 바탕으로 한 실무 추정치입니다.

## 핵심 기능

### 1. 족보 자동 매칭

`input/`의 오디오 파일명과 `jokbo/`의 PDF 파일명을 기준으로 날짜/교시/교수명을 맞춰 자동 매칭합니다.

예:

- `1교시 한정호 교수님.m4a`
- `20260319_1교시_신생아황달,대사장애_한정호 교수님_...pdf`

이 경우 PDF 파일명에서 강의명 `신생아황달,대사장애`도 자동으로 추출합니다.

### 2. 분할 파일 자동 그룹화

다음과 같은 파일은 같은 강의로 자동 묶입니다.

- `강의명_1.m4a`
- `강의명_2.m4a`
- `강의명 (1).m4a`
- `강의명 (2).m4a`

오디오는 각각 전사한 뒤 transcript를 합치고, 교정은 합쳐진 전체 텍스트를 한 번에 수행합니다.

### 3. 25MB 초과 파일 자동 처리

Whisper API 업로드 제한을 넘으면:

1. `--compress-for-api`가 있으면 경량 압축 시도
2. 그래도 크면 `ffmpeg`로 자동 분할
3. 분할 조각을 각각 전사
4. transcript를 합쳐서 한 번에 교정

즉 조각별 교정이 아니라 전체 맥락을 본 뒤 교정합니다.

### 4. 교정본 중심 저장

현재 주요 출력은 세 가지입니다.

- `*_transcript.txt`: Whisper raw transcript
- `*_corrected_transcript.txt`: LLM 교정본
- `logs/*_medical_map.json`: medical map 디버그

포맷 레이어를 켜지 않으면 `*_lecture_notes.md`는 만들지 않습니다.

### 5. 재처리 UX

이미 전사된 transcript는 Whisper를 다시 태우지 않고 교정만 다시 돌릴 수 있습니다.

```bash
python main.py --reprocess --correction-only --openrouter
```

특정 강의만 다시 교정하려면:

```bash
python main.py --reprocess --correction-only --openrouter --file "강의명_1.m4a"
```

이 경로가 가장 비용 효율적입니다.

## Medical Map 구조

`medical map`은 Whisper prompt와 correction layer에 넣는 의학용어 힌트 문자열입니다.

현재는 아래 신호를 조합합니다.

1. 족보 `급분바` 텍스트
2. `medical_dicts.py` 과목별/일반 의학용어 사전
3. prefix/suffix/abbreviation regex
4. allowlist / hard stopword / 저신호 약어 필터
5. 강의명 subject router 결과

### subject router

강의 제목은 작은 라우터 모델로 먼저 분류합니다.

- 기본 라우터 모델: `qwen/qwen3.5-9b`
- 역할: `["pediatrics"]`, `["obstetrics", "embryology"]` 같은 과목 태그만 반환
- 용어를 생성하지 않고, dict/scoring의 prior만 조정

이 구조 덕분에 제목 키워드를 코드에 하나씩 박는 방식을 줄이고, 과목별 dict를 더 넓게 써도 위험이 덜합니다.

### 현재 한계

- 족보에 영문 용어가 적으면 `medical map`이 약해질 수 있음
- raw STT가 크게 무너지면 correction layer도 복구 한계가 있음
- 따라서 큰 품질 향상은 코드보다 녹음 품질에서 나올 가능성이 큼

## 디렉터리 구조

```text
.
├── main.py
├── medical_dicts.py
├── requirements.txt
├── .env.template
├── backend.env.template
├── input/
├── jokbo/
├── output/
└── logs/
```

### 입력

- `input/`: `.m4a`, `.mp3`
- `jokbo/`: 족보 PDF

### 출력

- `output/*_transcript.txt`
- `output/*_corrected_transcript.txt`
- `output/*_lecture_notes.md` (선택)
- `logs/*.json`
- `logs/*_medical_map.json`

## 설치

### Python 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ffmpeg

분할/압축 기능에 필요합니다.

```bash
brew install ffmpeg
```

## 환경 변수

```bash
cp .env.template .env
cp backend.env.template backend.env
```

주요 값:

```dotenv
OPENAI_API_KEY=<OPENAI_API_KEY>
OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
HF_API_KEY=<HF_API_KEY>
HF_MODEL_ID=Qwen/Qwen2.5-72B-Instruct
SUBJECT_ROUTER_MODEL=qwen/qwen3.5-9b
```

현재 기본값:

- Whisper API: `whisper-1`
- OpenRouter correction model: `google/gemini-3.1-flash-lite-preview`
- subject router: `qwen/qwen3.5-9b`

## 자주 쓰는 명령

### 기본 추천

```bash
python main.py --correction-only --openrouter --compress-for-api
```

### 특정 강의만 처리

```bash
python main.py --correction-only --openrouter --file "강의명_1.m4a"
```

### medical map 디버그 같이 보기

```bash
python main.py --correction-only --openrouter --debug-medical-map --file "강의명.m4a"
```

### 기존 transcript만 다시 교정

```bash
python main.py --reprocess --correction-only --openrouter --file "강의명_1.m4a"
```

### 포맷 레이어까지 실행

```bash
python main.py --openrouter
```

기본 운영에서는 권장하지 않습니다. 정말 필요할 때만 사용합니다.

## CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--file` | 특정 강의만 처리. 관련 분할 파일 자동 포함 |
| `--whisper-only` | Whisper 전사만 수행 |
| `--correction-only` | 교정본까지만 저장 |
| `--reprocess` | 기존 transcript 재교정 |
| `--debug-medical-map` | medical map 후보/점수 디버그 저장 |
| `--compress-for-api` | 25MB 초과 시 압축을 먼저 시도 |
| `--lecture-title` | 강의명 수동 지정 |
| `--disable-subject-router` | 제목 라우터 비활성화 |
| `--subject-router-model` | 제목 라우터 모델 지정 |
| `--openrouter` | OpenRouter 사용 |
| `--local` | 로컬 Ollama 사용 |
| `--local-whisper` | 로컬 faster-whisper 사용 |
| `--whisper-model` | 로컬 Whisper 모델 크기 지정 |
| `--model` | correction/format 모델 수동 지정 |
| `--context-limit` | 컨텍스트 한계 지정 |
| `--use-reasoning` | OpenRouter reasoning 사용 |

## 로컬 Whisper에 대한 현재 판단

코드에는 로컬 Whisper 경로가 남아 있지만, 현재 운영 기준 기본 선택지는 아닙니다.

이유:

- Apple Silicon 환경에서 로컬 세팅이 API보다 UX가 나쁜 경우가 많음
- 긴 강의 파일은 메모리/시간 부담이 큼
- 현재는 `API Whisper + correction-only`가 더 안정적

즉 로컬 Whisper는 실험용 경로로 남아 있고, 기본 운영은 원격 API 기준입니다.

## 앞으로의 방향

당분간 큰 구조 개편보다 아래가 우선입니다.

- 녹음 품질 개선
- 과목별 dict 확장
- `medical_map.json`을 보며 allowlist/low-signal 약어 미세 조정
- 같은 transcript에 대한 `reprocess` 반복

지금 상태에서 가장 큰 품질 향상은 코드보다 입력 음질에서 나올 가능성이 큽니다.
