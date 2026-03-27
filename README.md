# Medical Lecture Processor

의학 강의 오디오를 전사하고, 족보 `급분바`와 강의명 prior를 이용해 STT 오인식을 보정한 뒤, 외부 LLM에 바로 넣기 좋은 `extracted notes`까지 뽑는 파이프라인입니다.

현재 이 프로젝트는 "자동 강의노트 생성기"보다 아래 두 산출물을 만드는 도구로 보는 편이 맞습니다.

- `*_corrected_transcript.txt`
- `*_extracted_notes.md`

## 운영 원칙

- 목표는 `환각 없이`, `원문에 가깝게`, `의학용어만 보정`하는 것
- 기본 운영은 `Whisper -> corrected transcript -> extracted notes`
- `study brief`는 기본 비활성화, 필요할 때만 옵션으로 생성
- 같은 transcript를 여러 번 튜닝할 수 있게 `reprocess` / `highlights-only` UX를 우선
- `medical map`은 교정 레이어에서만 적극 사용
- 하이라이트 레이어는 `강의명 + 제목 라우터가 잡은 큰 분과` 정도만 참고

## 현재 추천 플로우

### 1. 실사용 기본

```bash
python main.py --openrouter --compress-for-api
```

이 모드는 다음 순서로 동작합니다.

1. OpenAI Whisper API로 전사
2. 족보 PDF에서 `급분바` 섹션 추출
3. 강의명과 족보를 바탕으로 `medical map` 생성
4. OpenRouter LLM으로 STT 오인식만 교정
5. 교정본을 청크 단위로 다시 정제해 `extracted notes` 생성

기본 출력:

- `output/*_transcript.txt`
- `output/*_corrected_transcript.txt`
- `output/*_extracted_notes.md`
- `logs/*_chunk_highlights.jsonl`

### 2. 가장 싼 모드

```bash
python main.py --correction-only --openrouter --compress-for-api
```

이 모드는 `corrected transcript`까지만 만들고 하이라이트 레이어를 생략합니다.

### 3. 기존 교정본으로 extracted notes만 다시 만들기

```bash
python main.py --highlights-only --openrouter --file "강의명_1.m4a"
```

이 경로는 Whisper와 correction을 다시 태우지 않고, 기존 `*_corrected_transcript.txt`만 읽어서 `extracted notes`를 다시 생성합니다.

## 왜 이렇게 쓰는가

현재 경험상 병목은 점점 `예쁜 최종 노트 생성`보다 아래 두 곳에 있습니다.

- 원음 품질 / Whisper raw STT
- 교정본 이후의 과도한 압축

그래서 지금은 `final lecture note` 하나보다 아래 두 산출물이 더 실용적입니다.

- `corrected transcript`: 근거 보존용
- `extracted notes`: 외부 LLM 투입용

즉 "믿을 수 있는 교정본"과 "읽기 좋은 정제본"을 분리해서 쓰는 구조입니다.

## 비용 감각

기준선은 `Whisper API + correction-only + subject router`입니다.

- 50분 강의 1개: 약 `$0.33 ~ $0.35`
- 대부분은 Whisper 전사 비용
- correction layer는 보통 몇 센트 수준
- 제목 subject router 비용은 사실상 무시 가능

대략 구성:

- Whisper API: 50분 기준 약 `$0.30`
- correction layer: 보통 약 `$0.03 ~ $0.04`
- subject router: 거의 `0`

`extracted notes`까지 켜면 여기서 OpenRouter 청크 추출 비용이 추가됩니다. 따라서 가장 싼 운영은 여전히 `--correction-only`입니다.

## 핵심 기능

### 1. 족보 자동 매칭

`input/` 오디오 파일명과 `jokbo/` PDF 파일명을 기준으로 날짜/교시/교수명을 맞춰 자동 매칭합니다.

예:

- `1교시 한정호 교수님.m4a`
- `20260319_1교시_신생아황달,대사장애_한정호 교수님_...pdf`

이 경우 PDF 파일명에서 강의명 `신생아황달,대사장애`도 자동 추출합니다.

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

### 4. 듀얼 아웃풋

현재 실사용 기준 주요 출력은 두 가지입니다.

- `*_corrected_transcript.txt`: 최대한 원문에 충실한 교정본
- `*_extracted_notes.md`: 외부 LLM에 바로 넣기 좋은 정제본

보조 출력:

- `*_transcript.txt`: Whisper raw transcript
- `logs/*_chunk_highlights.jsonl`: 청크별 추출 원본
- `*_study_brief.md`: 옵션으로만 생성

### 5. 재처리 UX

이미 전사된 transcript는 Whisper를 다시 태우지 않고 교정만 다시 돌릴 수 있습니다.

```bash
python main.py --reprocess --correction-only --openrouter
```

기존 교정본에서 extracted notes만 다시 만들려면:

```bash
python main.py --highlights-only --openrouter --file "강의명_1.m4a"
```

이 두 경로가 비용 효율적입니다.

## Medical Map 구조

`medical map`은 Whisper prompt와 correction layer에 넣는 의학용어 힌트 문자열입니다.

현재는 아래 신호를 조합합니다.

1. 족보 `급분바` 텍스트
2. `medical_dicts.py` 과목별/일반 의학용어 사전
3. prefix/suffix/abbreviation regex
4. allowlist / hard stopword / 저신호 약어 필터
5. 강의명 subject router 결과

중요한 점:

- `medical map`은 correction 레이어에서 주로 사용
- `extracted notes` 하이라이트 레이어는 `medical map` 대신 `강의명 + 큰 분과`만 참고

즉 하이라이트 레이어는 원문과 모델 자체를 더 믿는 구조입니다.

## Subject Router

강의 제목은 작은 라우터 모델로 먼저 분류합니다.

- 기본 라우터 모델: `qwen/qwen3.5-9b`
- 역할: `["pediatrics"]`, `["obstetrics", "embryology"]` 같은 과목 태그만 반환
- 용어를 생성하지 않고, dict/scoring의 prior만 조정

하이라이트 레이어에서는 이 과목 태그를 거의 그대로 프롬프트에 넣습니다.

예:

- `Embryology, Histology, Obstetrics`
- `Anatomy, Gynecology`
- `Growth Development, Pediatrics`

## 하이라이트 레이어 구조

현재 하이라이트 레이어는 두 단계입니다.

1. `corrected transcript`를 청크로 나눔
2. 각 청크를 짧은 prose 노트로 정제

청크 추출은 아래 원칙을 따릅니다.

- 첫 줄은 짧은 소제목
- 그 아래 4~5줄 정도의 핵심 내용
- JSON 강제 없음
- 불확실하면 생략
- 장황한 요약보다 읽기 좋은 정제에 집중

즉 `extracted notes`는 "짧은 요약"이 아니라 "원문 기반 정제본"에 가깝습니다.

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

입력:

- `input/`: `.m4a`, `.mp3`
- `jokbo/`: 족보 PDF

출력:

- `output/*_transcript.txt`
- `output/*_corrected_transcript.txt`
- `output/*_extracted_notes.md`
- `output/*_study_brief.md` (옵션)
- `logs/*.json`
- `logs/*_medical_map.json`
- `logs/*_chunk_highlights.jsonl`

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
HIGHLIGHT_EXTRACT_MODEL=qwen/qwen3.5-9b
HIGHLIGHT_SYNTHESIS_MODEL=google/gemini-3.1-flash-lite-preview
```

현재 기본값:

- Whisper API: `whisper-1`
- OpenRouter correction model: `google/gemini-3.1-flash-lite-preview`
- subject router: `qwen/qwen3.5-9b`
- highlight extract model: `qwen/qwen3.5-9b`

## 자주 쓰는 명령

### 기본 추천

```bash
python main.py --openrouter --compress-for-api
```

### 특정 강의만 처리

```bash
python main.py --openrouter --file "강의명_1.m4a"
```

### correction-only

```bash
python main.py --correction-only --openrouter --file "강의명_1.m4a"
```

### 기존 transcript만 다시 교정

```bash
python main.py --reprocess --correction-only --openrouter --file "강의명_1.m4a"
```

### 기존 corrected transcript로 extracted notes만 다시 생성

```bash
python main.py --highlights-only --openrouter --file "강의명_1.m4a"
```

### medical map 디버그 같이 보기

```bash
python main.py --openrouter --debug-medical-map --file "강의명.m4a"
```

### 정말 필요할 때만 study brief 생성

```bash
python main.py --openrouter --study-brief --file "강의명_1.m4a"
```

## CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--file` | 특정 강의만 처리. 관련 분할 파일 자동 포함 |
| `--whisper-only` | Whisper 전사만 수행 |
| `--correction-only` | 교정본까지만 저장 |
| `--highlights-only` | 기존 corrected transcript에서 extracted notes만 생성 |
| `--study-brief` | 최종 study brief까지 추가 생성 |
| `--reprocess` | 기존 transcript 재교정 |
| `--debug-medical-map` | medical map 후보/점수 디버그 저장 |
| `--compress-for-api` | 25MB 초과 시 압축을 먼저 시도 |
| `--lecture-title` | 강의명 수동 지정 |
| `--disable-subject-router` | 제목 라우터 비활성화 |
| `--subject-router-model` | 제목 라우터 모델 지정 |
| `--highlight-extract-model` | extracted notes용 청크 추출 모델 지정 |
| `--highlight-synthesis-model` | study brief 통합 모델 지정 |
| `--highlight-chunk-tokens` | 하이라이트 청크 크기 지정 |
| `--highlight-overlap-tokens` | 하이라이트 청크 overlap 지정 |
| `--openrouter` | OpenRouter 사용 |
| `--local` | 로컬 Ollama 사용 |
| `--local-whisper` | 로컬 faster-whisper 사용 |
| `--whisper-model` | 로컬 Whisper 모델 크기 지정 |
| `--model` | correction 모델 수동 지정 |
| `--context-limit` | 컨텍스트 한계 지정 |
| `--use-reasoning` | OpenRouter reasoning 사용 |

## 로컬 Whisper에 대한 현재 판단

코드에는 로컬 Whisper 경로가 남아 있지만, 현재 운영 기준 기본 선택지는 아닙니다.

이유:

- Apple Silicon 환경에서 로컬 세팅이 API보다 UX가 나쁜 경우가 많음
- 긴 강의 파일은 메모리/시간 부담이 큼
- 현재는 `API Whisper + correction + extracted notes`가 더 안정적

즉 로컬 Whisper는 실험용 경로로 남아 있고, 기본 운영은 원격 API 기준입니다.

## 앞으로의 방향

당분간 큰 구조 개편보다 아래가 우선입니다.

- 녹음 품질 개선
- 과목별 dict 확장
- `medical_map.json`을 보며 allowlist/low-signal 약어 미세 조정
- `corrected transcript + extracted notes` 조합을 계속 다듬기
- extracted notes의 비용/속도/보존력 최적화

지금 상태에서 가장 큰 품질 향상은 코드보다 입력 음질에서 나올 가능성이 큽니다.
