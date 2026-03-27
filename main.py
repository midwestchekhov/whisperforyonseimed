import os
import re
import json
import time
import argparse
import unicodedata
from datetime import datetime
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
import requests
from medical_dicts import GENERAL_MEDICAL_TERMS, SUBJECT_SPECIFIC_TERMS

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


# Load environment variables
load_dotenv(dotenv_path=".env", override=True)
load_dotenv(dotenv_path="backend.env", override=True)

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
HF_API_KEY         = os.getenv("HF_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

HF_MODEL_ID    = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
MEDICAL_KEYWORDS = os.getenv(
    "MEDICAL_KEYWORDS",
    "의학, 해부학, 생리학, 병리학, 약리학, 의대, 교수님, 중간고사, 기말고사"
)

# Initialize OpenAI client for Whisper
whisper_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize Hugging Face Router client
if HF_API_KEY:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_API_KEY,
        timeout=180.0
    )
    RELIABLE_HF_MODEL_ID = f"{HF_MODEL_ID}:novita"
else:
    hf_client = None

# Initialize OpenRouter client
if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        timeout=180.0,
        default_headers={
            "HTTP-Referer": "https://github.com/midwestchekhov/Whisper-API-test",
            "X-OpenRouter-Title": "Medical Lecture Processor",
        }
    )
    DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-flash-lite-preview"
    DEFAULT_SUBJECT_ROUTER_MODEL = os.getenv("SUBJECT_ROUTER_MODEL", "qwen/qwen3.5-9b")
    DEFAULT_HIGHLIGHT_EXTRACT_MODEL = os.getenv("HIGHLIGHT_EXTRACT_MODEL", "qwen/qwen3.5-9b")
    DEFAULT_HIGHLIGHT_SYNTHESIS_MODEL = os.getenv("HIGHLIGHT_SYNTHESIS_MODEL", DEFAULT_OPENROUTER_MODEL)
else:
    openrouter_client = None
DEFAULT_SUBJECT_ROUTER_MODEL = "qwen/qwen3.5-9b"
DEFAULT_HIGHLIGHT_EXTRACT_MODEL = "qwen/qwen3.5-9b"
DEFAULT_HIGHLIGHT_SYNTHESIS_MODEL = "google/gemini-3.1-flash-lite-preview"

# Initialize local Ollama client
OLLAMA_CLIENT = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Directories
INPUT_DIR  = Path("input")
OUTPUT_DIR = Path("output")
LOG_DIR    = Path("logs")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
SUBJECT_ROUTER_CACHE_PATH = LOG_DIR / "subject_router_cache.json"

JOKBO_DIR = Path("jokbo")
JOKBO_DIR.mkdir(exist_ok=True)

# ── 급분바 섹션 검색 패턴 ────────────────────────────────────────────────────
PAT_START = re.compile(r"급\s*분\s*바", re.IGNORECASE)
PAT_END   = re.compile(r"문\s*족",      re.IGNORECASE)

# ── 영문 의학용어 regex (Whisper prompt 우선순위 추출용) ───────────────────────
# ※ \b 대신 (?<![A-Za-z]) / (?![A-Za-z]) 사용
#    → Python 3에서 \w가 한글을 포함하므로 \b가 한-영 경계에서 동작하지 않는 문제 수정
MEDICAL_PREFIX_RE = re.compile(
    r'(?<![A-Za-z])(anti|hyper|hypo|pre|post|sub|inter|intra|de|brady|tachy|poly|mono|'
    r'micro|macro|pseudo|neo|pan|hemi)[a-z]{3,}(?![A-Za-z])',
    re.IGNORECASE
)
MEDICAL_SUFFIX_RE = re.compile(
    r'(?<![A-Za-z])[a-z]{3,}(itis|oma|osis|pathy|logy|ectomy|plasty|scopy|tomy|stomy|'
    r'graphy|gram|emia|uria|algia|trophy|genesis|plasia|megaly)(?![A-Za-z])',
    re.IGNORECASE
)
# 순수 대문자(ECG, MI, SGLT2) + 소문자 혼용 약어(HbA1c, eGFR, pH, mRNA)
ABBREV_RE = re.compile(
    r'(?<![A-Za-z])'
    r'(?:[A-Z]{2,}[0-9]*|[A-Za-z][a-z0-9]*[A-Z][A-Za-z0-9]*)'
    r'(?![A-Za-z0-9])'
)
GROUP_SUFFIX_RE = re.compile(r'(_\d+|\s*\(\d+\))$')

# medical map 정제는 regex 하나로 완성하기보다,
# 고신뢰 사전 힌트와 소수의 hard-stop/allowlist를 함께 쓰는 쪽이 더 안정적이다.
MEDICAL_MAP_ALLOWLIST = {
    "ENGAGEMENT",
    "LABOR",
    "FRACTURE",
}
MEDICAL_MAP_HARD_STOPWORDS = {
    "PRESENT",
    "PRESENTED",
    "DEFINED",
    "DERIVED",
    "DEVELOPED",
    "PREDOMINANCE",
    "MICROSCOPIC",
    "MACROSCOPIC",
    "SUBUNIT",
    "DEFECT",
    "DEPENDENT",
    "DIAGNOSIS",
}
LOW_SIGNAL_ABBREVIATIONS = {
    "HR",
    "RR",
    "BP",
    "SS",
    "SD",
    "BV",
    "MF",
    "ML",
    "SE",
    "TA",
    "II",
    "CV",
    "CP",
    "TZ",
    "BL",
    "DC",
    "IS",
    "PM",
    "SK",
}

def _is_english_medical_term(term: str) -> bool:
    return bool(re.search(r"[A-Za-z]", term)) and not bool(re.search(r"[가-힣]", term))


TERM_TO_SUBJECTS: dict[str, set[str]] = {}
for subject, terms in SUBJECT_SPECIFIC_TERMS.items():
    for term in terms:
        normalized_term = unicodedata.normalize("NFC", term.strip())
        if normalized_term and _is_english_medical_term(normalized_term):
            TERM_TO_SUBJECTS.setdefault(normalized_term.upper(), set()).add(subject)

MEDICAL_DICT_TERMS = sorted(
    {
        unicodedata.normalize("NFC", term.strip())
        for term in GENERAL_MEDICAL_TERMS
        + [item for terms in SUBJECT_SPECIFIC_TERMS.values() for item in terms]
        if term.strip() and _is_english_medical_term(term)
    },
    key=lambda term: (-len(term), term.lower()),
)
MEDICAL_DICT_PATTERNS = [
    (
        term,
        re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    )
    for term in MEDICAL_DICT_TERMS
]
LECTURE_TITLE_KEYWORD_GROUPS: dict[str, set[str]] = {
    "조직학": {"histology"},
    "histology": {"histology"},
    "발생": {"embryology"},
    "배아": {"embryology"},
    "embryology": {"embryology"},
    "해부": {"anatomy"},
    "anatomy": {"anatomy"},
    "생리": {"physiology"},
    "physiology": {"physiology"},
    "병리": {"pathology", "oncology"},
    "pathology": {"pathology", "oncology"},
    "약리": {"pharmacology"},
    "pharmacology": {"pharmacology"},
    "면역": {"immunology"},
    "immunology": {"immunology"},
    "산과": {"obstetrics"},
    "ob": {"obstetrics"},
    "부인": {"gynecology"},
    "gyn": {"gynecology"},
    "여성생식": {"gynecology", "obstetrics", "histology", "anatomy"},
    "난소": {"gynecology", "obstetrics", "histology"},
    "자궁": {"gynecology", "obstetrics", "histology"},
    "난관": {"gynecology", "obstetrics", "histology"},
    "소아": {"pediatrics"},
    "pedi": {"pediatrics"},
    "신생아": {"pediatrics"},
    "태아": {"obstetrics", "embryology", "histology", "anatomy"},
    "태반": {"obstetrics", "embryology", "histology"},
    "착상": {"obstetrics", "embryology", "histology"},
    "수정": {"embryology"},
    "진찰": {"pediatrics"},
    "관리": {"pediatrics"},
    "성장": {"pediatrics", "physiology"},
    "발달": {"pediatrics", "physiology", "obstetrics", "embryology"},
    "종양": {"oncology", "pathology"},
    "oncology": {"oncology", "pathology"},
    "reproductive": {"gynecology", "obstetrics", "histology", "anatomy"},
}
SUBJECT_ROUTER_SUBJECTS = tuple(sorted(SUBJECT_SPECIFIC_TERMS.keys()))

# ── 교정 레이어 프롬프트 ─────────────────────────────────────────────────────
CORRECTION_SYSTEM = (
    "당신은 의학 강의 STT(음성인식) 교정 전문가입니다.\n"
    "족보(시험 출제 자료)를 레퍼런스로 삼아 전사문의 오인식만 교정합니다.\n\n"
    "교정 대상:\n"
    "  - 의학 외래어를 한글로 잘못 전사한 것 (예: 페라티노사이트 → keratinocyte)\n"
    "  - 영문 약어/용어 오인식 (예: IR17 → IL-17)\n"
    "  - 유사 발음 한국어 오인식 (예: 도시검사 → 조직검사)\n\n"
    "규칙:\n"
    "  1. 족보에 나오는 용어를 최우선 레퍼런스로 사용하세요.\n"
    "  2. 불확실하면 원문 그대로 두세요. 과교정 금지.\n"
    "  3. 오인식 단어 교정만 하고, 문장 구조·내용·말투는 절대 바꾸지 마세요.\n"
    "  4. 교정된 전사문만 출력하세요 (설명·주석·인사말 없이)."
)

# 엔드포인트 컨텍스트 한계 (입출력 합산, Gemini 등 현대 모델 대응)
CONTEXT_LIMIT = 200000
HIGHLIGHT_EXTRACT_MAX_TOKENS = 700
HIGHLIGHT_SYNTHESIS_MAX_TOKENS = 6000


def _safe_max_tokens(system_msg: str, user_msg: str, desired: int = 16384) -> int:
    """
    입력 크기를 추정해 컨텍스트 한계를 초과하지 않는 max_tokens 반환.
    한국어 혼합 텍스트 기준 약 1.1자 ≈ 1토큰 (보수적 상한 추정).
    """
    estimated_input = int((len(system_msg) + len(user_msg)) / 1.1)
    
    # 입력이 매우 큰 경우, 출력 공간도 그에 맞춰 확보 시도
    # Gemini 3.1 Flash 등은 context가 매우 크므로 보수적인 desired를 상황에 맞게 상기
    adjusted_desired = desired
    if estimated_input > (desired * 0.8): # 입력이 어느 정도 크면 출력 공간을 더 넉넉히 잡음
        adjusted_desired = min(128000, estimated_input + 10000) 

    safe = CONTEXT_LIMIT - estimated_input - 2000  # 2000토큰 여유 버퍼
    result = max(4096, min(adjusted_desired, safe))
    
    if result < adjusted_desired:
        print(f"  ⚠️  컨텍스트 초과 방지: max_tokens 시도 {adjusted_desired} → {result} (입력 추정 {estimated_input:,}토큰)")
    return result


def _bounded_max_tokens(
    system_msg: str,
    user_msg: str,
    *,
    desired: int,
    floor: int,
) -> int:
    """
    추출/통합 레이어처럼 출력량이 비교적 예측 가능한 호출에 쓰는 보수 상한.
    `_safe_max_tokens`처럼 출력 한도를 과도하게 키우지 않는다.
    """
    estimated_input = int((len(system_msg) + len(user_msg)) / 1.1)
    safe = CONTEXT_LIMIT - estimated_input - 2000
    if safe <= 0:
        return floor
    result = min(desired, safe)
    return max(floor, result)


# ════════════════════════════════════════════════════════════════════════════
#   PDF 추출: 급분바 섹션만
# ════════════════════════════════════════════════════════════════════════════

def is_black_color(color) -> bool:
    """PyMuPDF span color가 검정인지 판단 (int packed RGB 또는 tuple 모두 처리)."""
    if isinstance(color, (int, float)):
        v = int(color)
        r = (v >> 16) & 0xFF
        g = (v >>  8) & 0xFF
        b =  v        & 0xFF
        return r < 30 and g < 30 and b < 30
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return color[0] < 30 and color[1] < 30 and color[2] < 30
    return False


def find_best_label_page(doc, pattern, start_page: int = 0, end_page: int = None):
    """
    doc 내 start_page~end_page 범위에서 pattern을 포함하는
    '검정색 + 가장 큰 폰트' span의 페이지 인덱스를 반환.
    못 찾으면 (None, -1) 반환.
    """
    if end_page is None or end_page > doc.page_count:
        end_page = doc.page_count

    best_page = None
    best_size = -1.0

    for i in range(start_page, end_page):
        page = doc.load_page(i)
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "") or ""
                    txt = unicodedata.normalize("NFC", txt)
                    if not pattern.search(txt):
                        continue
                    color = span.get("color", 0)
                    if not is_black_color(color):
                        continue
                    size = float(span.get("size", 0.0))
                    if size > best_size:
                        best_size = size
                        best_page = i

    return best_page, best_size


def _extract_all_pages(doc) -> str:
    """fallback: 전체 페이지 텍스트 추출."""
    pages = []
    for i in range(doc.page_count):
        pages.append(unicodedata.normalize("NFC", doc[i].get_text("text")))
    return "\n".join(pages).strip()


import subprocess

def check_ffmpeg_available() -> bool:
    """ ffmpeg가 시스템에 설치되어 있는지 확인. """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def split_audio_ffmpeg(audio_path: Path, output_dir: Path, segment_mb: int = 24) -> list[Path]:
    """
    ffmpeg를 사용하여 오디오 파일을 지정된 용량(약 segment_mb) 기준으로 분할.
    OpenAI 25MB 제한을 피하기 위해 기본 24MB로 설정.
    """
    if not check_ffmpeg_available():
        print("❌  ffmpeg가 설치되어 있지 않습니다. (macOS: brew install ffmpeg)")
        return []

    print(f"✂️   파일 분할 중 (25MB 초과): {audio_path.name}")
    
    # 임시 폴더 생성
    split_dir = output_dir / f"splits_{audio_path.stem}"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # ffmpeg segment 기능을 사용하여 분할
    # -f segment -segment_time 600 (10분씩 분할은 용량 가늠이 어려우므로 
    # 간단하게 용량 기반으로 나누기 위해 bitrate를 고려한 시간 계산이 필요할 수 있으나,
    # 여기서는 안전하게 15분(900초) 단위로 분할 시도)
    output_pattern = str(split_dir / f"{audio_path.stem}_part%03d{audio_path.suffix}")
    
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", "900", # 15분 단위 (일반적인 압축률에서 25MB 미만 유지 확률 높음)
        "-c", "copy",
        output_pattern
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        splits = sorted(list(split_dir.glob(f"*{audio_path.suffix}")))
        print(f"✅  {len(splits)}개의 조각으로 분할 완료.")
        return splits
    except Exception as e:
        print(f"❌  파일 분할 오류: {e}")
        return []


def compress_audio_ffmpeg(audio_path: Path, output_dir: Path,
                          sample_rate: int = 16000, bitrate: str = "40k") -> Path | None:
    """
    Whisper API 업로드 전 경량 재인코딩.
    mono + 저샘플레이트 + 낮은 AAC bitrate로 용량을 줄인다.
    """
    if not check_ffmpeg_available():
        print("❌  ffmpeg가 설치되어 있지 않습니다. (macOS: brew install ffmpeg)")
        return None

    compressed_path = output_dir / f"{audio_path.stem}_compressed.m4a"
    print(f"🗜️   파일 압축 시도 중: {audio_path.name} -> {compressed_path.name}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "aac",
        "-b:a", bitrate,
        str(compressed_path),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        compressed_mb = compressed_path.stat().st_size / (1024 * 1024)
        print(f"✅  압축 완료: {compressed_path.name} ({compressed_mb:.1f}MB)")
        return compressed_path
    except Exception as e:
        print(f"❌  파일 압축 오류: {e}")
        return None


def extract_jukbo_section(pdf_path: Path) -> str:
    """
    족보 PDF에서 '급분바' ~ '문족' 사이 섹션만 추출.
    급분바 라벨 페이지를 찾지 못하면 전체 페이지를 fallback으로 반환.
    """
    if fitz is None:
        print("❌  PyMuPDF(fitz) 미설치: pip install pymupdf")
        return ""
    print(f"📄  PDF 급분바 섹션 추출 중: '{pdf_path.name}'...")
    try:
        doc = fitz.open(str(pdf_path))
        start_page, _ = find_best_label_page(doc, PAT_START)

        if start_page is None:
            print(f"⚠️   '{pdf_path.name}': '급분바' 라벨 페이지를 찾지 못했습니다 → 전체 페이지 fallback.")
            return _extract_all_pages(doc)

        print(f"✅  '급분바' 섹션 시작: {start_page + 1}페이지")
        collected = []

        for page_num in range(start_page, doc.page_count):
            page = doc.load_page(page_num)
            page_text = unicodedata.normalize("NFC", page.get_text("text"))

            if page_num > start_page and PAT_END.search(page_text):
                # '문족' 이전까지만 수집
                m = PAT_END.search(page_text)
                collected.append(page_text[:m.start()])
                print(f"✅  '문족' 발견 → {page_num + 1}페이지에서 종료.")
                break
            collected.append(page_text)

        result = "\n".join(collected).strip()
        print(f"✅  급분바 섹션 추출 완료: {len(result)}자")
        return result

    except Exception as e:
        print(f"❌  PDF 읽기 오류 ({pdf_path.name}): {e}")
        return ""


# ════════════════════════════════════════════════════════════════════════════
#   자동 매칭 유틸리티 (Date + Professor Name)
# ════════════════════════════════════════════════════════════════════════════

def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def extract_match_keys(filename: str) -> tuple[str, str, str]:
    """
    파일명에서 매칭 키(날짜, 교시, 교수명) 추출.
    양식:
    - 오디오: YYYYMMDD_#교시_교수명.m4a
    - 족보: YYYYMMDD_#교시_수업명_교수명_작성자명_수정(#).pdf
    - fallback: #교시_교수명.pdf / '#교시 교수명 교수님_1.m4a' 같은 단순 이름도 허용
    """
    name = group_audio_stem(normalize_nfc(Path(filename).stem))
    parts = [p.strip() for p in name.split("_")]

    # 1. 날짜 추출 (8자리 숫자)
    date_part = ""
    if len(parts) > 0 and re.match(r"^\d{8}", parts[0]):
        date_part = parts[0][:8]

    # 2. 교시 추출
    session_part = ""
    session_match = re.search(r"(\d+)\s*교시", name)
    if session_match:
        session_part = f"{session_match.group(1)}교시"

    # 3. 교수명 추출
    prof_name = ""

    prof_matches = re.findall(r"([가-힣A-Za-z]+)\s*교수님?", name)
    if prof_matches:
        prof_name = prof_matches[-1].strip()

    if not prof_name:
        # '교수' 키워드가 없어도 '#교시_정호성.pdf' 같은 형태를 처리
        candidate_parts = parts[1:] if len(parts) > 1 else parts
        for part in candidate_parts:
            clean = part
            clean = re.sub(r"^\d{8}", "", clean).strip()
            clean = re.sub(r"\d+\s*교시", "", clean).strip()
            clean = clean.replace("교수님", "").replace("교수", "").strip(" _-")
            if not clean or clean.lower().startswith("pages from") or clean == "dummy":
                continue
            prof_name = clean
            break

    return date_part, session_part, prof_name


def extract_lecture_title(filename: str) -> str:
    """
    파일명에서 강의명(있다면)을 추출.
    예:
    - 20260324_4교시_미토콘드리아 유전과 에너지 대사질환_이영목교수님_...
      -> 미토콘드리아 유전과 에너지 대사질환
    - 1교시_정호성 / 1교시 정호성 교수님_1
      -> ""
    """
    name = group_audio_stem(normalize_nfc(Path(filename).stem))
    if name.lower().startswith("pages from "):
        name = name[len("Pages from "):].strip()
    parts = [p.strip() for p in name.split("_") if p.strip()]

    if len(parts) >= 4 and re.match(r"^\d{8}$", parts[0]) and re.search(r"\d+\s*교시", parts[1]):
        title = parts[2].strip()
        if "교수" not in title and title.lower() != "dummy":
            return title

    return ""


def infer_title_subjects(lecture_title: str) -> set[str]:
    normalized_title = normalize_nfc(lecture_title).lower()
    inferred: set[str] = set()
    for keyword, subjects in LECTURE_TITLE_KEYWORD_GROUPS.items():
        if keyword.lower() in normalized_title:
            inferred.update(subjects)
    return inferred


def build_highlight_subject_hint(subjects: set[str] | list[str] | tuple[str, ...]) -> str:
    normalized = sorted(
        {
            str(item).strip().lower().replace("-", " ")
            for item in subjects
            if str(item).strip()
        }
    )
    return ", ".join(word.title() for word in normalized)


def _load_subject_router_cache() -> dict[str, dict]:
    if not SUBJECT_ROUTER_CACHE_PATH.exists():
        return {}
    try:
        with open(SUBJECT_ROUTER_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_subject_router_cache(cache: dict[str, dict]) -> None:
    try:
        with open(SUBJECT_ROUTER_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _extract_json_object(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def route_title_subjects(
    lecture_title: str,
    *,
    model: str = DEFAULT_SUBJECT_ROUTER_MODEL,
    use_router: bool = True,
) -> dict:
    keyword_subjects = sorted(infer_title_subjects(lecture_title))
    normalized_title = normalize_nfc(lecture_title).strip()
    fallback = {
        "subjects": keyword_subjects,
        "source": "keyword-fallback" if normalized_title else "empty-title",
        "model": model,
        "confidence": 0.0,
        "reason": "",
        "keyword_subjects": keyword_subjects,
    }

    if not normalized_title:
        return fallback
    if not use_router or not openrouter_client:
        return fallback

    cache_key = f"v5::{model}::{normalized_title}"
    cache = _load_subject_router_cache()
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        cached["keyword_subjects"] = keyword_subjects
        cached.setdefault("model", model)
        return cached

    user_msg = (
        "당신은 의학 강의 제목 분류기입니다.\n"
        f"강의 제목: {normalized_title}\n"
        f"허용 태그: {', '.join(SUBJECT_ROUTER_SUBJECTS)}\n"
        "규칙:\n"
        "- 허용 태그 중 0~3개만 고르기\n"
        "- 제목만으로 추정하기\n"
        "- 불확실하면 적게 고르기\n"
        "- 새 태그를 만들지 않기\n"
        "- JSON만 출력하기\n"
        '{"subjects":["..."],"confidence":0.0,"reason":"짧게"}'
    )

    try:
        completion = openrouter_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0,
            max_tokens=160,
            extra_body={"reasoning": {"effort": "none", "exclude": True}},
        )
        content = completion.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        subjects: list[str] = []
        confidence = 0.0
        reason = ""
        if parsed:
            raw_subjects = parsed.get("subjects", [])
            if isinstance(raw_subjects, list):
                subjects = [
                    str(item) for item in raw_subjects
                    if str(item) in SUBJECT_ROUTER_SUBJECTS
                ]
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            reason = str(parsed.get("reason", "")).strip()

        result = {
            "subjects": sorted(dict.fromkeys(subjects))[:3] or keyword_subjects,
            "source": "llm" if subjects else "keyword-fallback",
            "model": model,
            "confidence": confidence,
            "reason": reason,
            "keyword_subjects": keyword_subjects,
        }
    except Exception as exc:
        result = {
            **fallback,
            "source": "keyword-fallback",
            "reason": f"router-error:{type(exc).__name__}",
        }

    cache[cache_key] = result
    _save_subject_router_cache(cache)
    return result


def find_matching_jokbo(audio_filename: str, jokbo_dir: Path) -> Path | None:
    """ audio_filename과 날짜/교시/교수명이 가장 잘 맞는 jokbo PDF 탐색. """
    a_date, a_session, a_prof = extract_match_keys(audio_filename)
    if not a_prof:
        return None

    print(f"🔍  매칭 키 추출: 날짜={a_date or '-'}, 교시={a_session or '-'}, 교수명={a_prof}")

    best_match: Path | None = None
    best_score = -1

    for pdf in jokbo_dir.glob("*.pdf"):
        p_date, p_session, p_prof = extract_match_keys(pdf.name)

        if not p_prof:
            continue
        if not (a_prof in p_prof or p_prof in a_prof):
            continue
        if a_date and p_date and a_date != p_date:
            continue
        if a_session and p_session and a_session != p_session:
            continue

        score = 0
        if a_prof == p_prof:
            score += 4
        else:
            score += 3
        if a_session and p_session and a_session == p_session:
            score += 2
        if a_date and p_date and a_date == p_date:
            score += 3

        if score > best_score:
            best_match = pdf
            best_score = score

    if best_match is not None:
        print(f"🎯  매칭 성공: '{best_match.name}'")
    return best_match


# ════════════════════════════════════════════════════════════════════════════
#   키워드 추출 (Whisper prompt용)
# ════════════════════════════════════════════════════════════════════════════

def extract_medical_keywords(text: str) -> str:
    """
    급분바 텍스트에서 Whisper initial_prompt 용 키워드 추출.
    우선순위: 영문 접두사패턴 > 영문 접미사패턴 > 대문자 약어 > 한글 고빈도어.
    상위 40개를 콤마 구분 문자열로 반환.
    """
    return analyze_medical_keywords(text)["final_prompt"]


def _collect_match_debug(pattern: re.Pattern, text: str) -> tuple[list[str], list[dict[str, int | str]]]:
    """정규식 매치의 등장 순서와 등장 횟수를 함께 수집."""
    seen_order: list[str] = []
    first_seen: dict[str, str] = {}
    counts: Counter[str] = Counter()

    for m in pattern.finditer(text):
        term = m.group()
        key = term.upper()
        counts[key] += 1
        if key not in first_seen:
            first_seen[key] = term
            seen_order.append(key)

    ordered_terms = [first_seen[key] for key in seen_order]
    ordered_counts = [
        {"term": first_seen[key], "count": counts[key]}
        for key in seen_order
    ]
    return ordered_terms, ordered_counts


def _collect_dictionary_hits(text: str) -> tuple[list[str], list[dict[str, int | str]]]:
    """의학용어 사전 exact/phrase hit를 수집."""
    normalized = unicodedata.normalize("NFC", text)
    lowered = normalized.lower()
    ordered_terms: list[str] = []
    ordered_counts: list[dict[str, int | str]] = []

    for term, pattern in MEDICAL_DICT_PATTERNS:
        if term.lower() not in lowered:
            continue
        matches = list(pattern.finditer(normalized))
        if not matches:
            continue
        ordered_terms.append(term)
        ordered_counts.append({"term": term, "count": len(matches)})

    return ordered_terms, ordered_counts


def _score_medical_candidate(
    term: str,
    source_counts: dict[str, int],
    primary_title_subjects: set[str] | None = None,
    secondary_title_subjects: set[str] | None = None,
) -> tuple[int, list[str]]:
    key = term.upper()
    total_occurrences = sum(source_counts.values())
    score = 0
    reasons: list[str] = []
    primary_title_subjects = primary_title_subjects or set()
    secondary_title_subjects = secondary_title_subjects or set()

    if key in MEDICAL_MAP_HARD_STOPWORDS and key not in MEDICAL_MAP_ALLOWLIST:
        score -= 100
        reasons.append("hard-stop")

    if "dictionary" in source_counts:
        dict_score = 8 if (" " in term or "-" in term) else 6
        if key.isupper() and len(re.sub(r"[^A-Z0-9]", "", key)) <= 3:
            dict_score = 3
        score += dict_score
        reasons.append(f"dictionary:+{dict_score}")

    if "abbreviation" in source_counts:
        abbrev_score = 4
        if re.search(r"\d", term):
            abbrev_score += 2
        if "-" in term:
            abbrev_score += 1
        if re.search(r"[a-z]", term) and re.search(r"[A-Z]", term):
            abbrev_score += 1
        score += abbrev_score
        reasons.append(f"abbreviation:+{abbrev_score}")

    if "suffix" in source_counts:
        suffix_score = 3 if source_counts["suffix"] > 1 else 2
        score += suffix_score
        reasons.append(f"suffix:+{suffix_score}")

    if "prefix" in source_counts:
        prefix_score = 2 if source_counts["prefix"] > 1 else 1
        score += prefix_score
        reasons.append(f"prefix:+{prefix_score}")

    occurrence_bonus = min(total_occurrences, 3)
    if occurrence_bonus:
        score += occurrence_bonus
        reasons.append(f"occurrences:+{occurrence_bonus}")

    if key in MEDICAL_MAP_ALLOWLIST:
        score += 8
        reasons.append("allowlist:+8")

    subject_hits = TERM_TO_SUBJECTS.get(key, set())
    all_title_subjects = primary_title_subjects | secondary_title_subjects
    if all_title_subjects and subject_hits:
        primary_overlap = sorted(subject_hits & primary_title_subjects)
        secondary_overlap = sorted(subject_hits & secondary_title_subjects)
        if primary_overlap:
            title_score = 4 + min(len(primary_overlap), 2)
            if "dictionary" in source_counts:
                title_score += 2
            score += title_score
            reasons.append(f"subject-match({','.join(primary_overlap)}):+{title_score}")
        elif secondary_overlap:
            title_score = 2 + min(len(secondary_overlap), 1)
            if "dictionary" in source_counts:
                title_score += 1
            score += title_score
            reasons.append(f"keyword-match({','.join(secondary_overlap)}):+{title_score}")
        elif len(subject_hits) == 1:
            mismatch_penalty = 2 if "dictionary" in source_counts else 1
            score -= mismatch_penalty
            reasons.append(
                f"subject-mismatch({','.join(sorted(subject_hits))}):-{mismatch_penalty}"
            )

    if key in LOW_SIGNAL_ABBREVIATIONS:
        score -= 9
        reasons.append("low-signal-abbrev:-9")

    if "dictionary" not in source_counts and term.islower():
        score -= 2
        reasons.append("plain-lowercase:-2")
        if len(source_counts) == 1 and total_occurrences == 1:
            score -= 2
            reasons.append("single-regex-hit:-2")

    return score, reasons


def analyze_medical_keywords(
    text: str,
    lecture_title: str = "",
    *,
    subject_router_model: str = DEFAULT_SUBJECT_ROUTER_MODEL,
    use_subject_router: bool = True,
) -> dict:
    """
    medical map 추출 결과를 디버그 가능한 구조로 반환.
    어떤 카테고리에서 어떤 순서로 후보가 들어왔는지 추적하는 용도.
    """
    normalized = unicodedata.normalize("NFC", text)
    keyword_title_subjects = infer_title_subjects(lecture_title)
    subject_router = route_title_subjects(
        lecture_title,
        model=subject_router_model,
        use_router=use_subject_router,
    )
    primary_title_subjects = set(subject_router["subjects"])
    secondary_title_subjects = set(keyword_title_subjects) - primary_title_subjects
    title_subjects = primary_title_subjects | secondary_title_subjects

    dict_matches, dict_occurrences = _collect_dictionary_hits(normalized)
    prefix_matches, prefix_occurrences = _collect_match_debug(MEDICAL_PREFIX_RE, normalized)
    suffix_matches, suffix_occurrences = _collect_match_debug(MEDICAL_SUFFIX_RE, normalized)
    abbrev_matches, abbrev_occurrences = _collect_match_debug(ABBREV_RE, normalized)

    occurrence_lookup: dict[str, dict[str, int]] = {
        "dictionary": {item["term"].upper(): int(item["count"]) for item in dict_occurrences},
        "prefix": {item["term"].upper(): int(item["count"]) for item in prefix_occurrences},
        "suffix": {item["term"].upper(): int(item["count"]) for item in suffix_occurrences},
        "abbreviation": {item["term"].upper(): int(item["count"]) for item in abbrev_occurrences},
    }

    candidate_map: dict[str, dict[str, object]] = {}

    def _merge_candidate(term: str, source: str):
        key = term.upper()
        candidate = candidate_map.setdefault(
            key,
            {
                "term": term,
                "source_counts": {},
            },
        )
        source_counts = candidate["source_counts"]
        assert isinstance(source_counts, dict)
        source_counts[source] = occurrence_lookup[source][key]

        chosen_term = candidate["term"]
        if source == "dictionary" or len(term) > len(str(chosen_term)):
            candidate["term"] = term

    for term in dict_matches:
        _merge_candidate(term, "dictionary")
    for term in prefix_matches:
        _merge_candidate(term, "prefix")
    for term in suffix_matches:
        _merge_candidate(term, "suffix")
    for term in abbrev_matches:
        _merge_candidate(term, "abbreviation")

    scored_candidates: list[dict[str, object]] = []
    for key, candidate in candidate_map.items():
        term = str(candidate["term"])
        source_counts = dict(candidate["source_counts"])
        score, reasons = _score_medical_candidate(
            term,
            source_counts,
            primary_title_subjects=primary_title_subjects,
            secondary_title_subjects=secondary_title_subjects,
        )
        total_occurrences = sum(source_counts.values())
        scored_candidates.append({
            "term": term,
            "score": score,
            "reasons": reasons,
            "sources": sorted(source_counts.keys()),
            "occurrences": source_counts,
            "total_occurrences": total_occurrences,
            "allowlisted": key in MEDICAL_MAP_ALLOWLIST,
            "subjects": sorted(TERM_TO_SUBJECTS.get(key, set())),
        })

    scored_candidates.sort(
        key=lambda item: (
            -int(item["score"]),
            -int(item["total_occurrences"]),
            item["term"].lower(),
        )
    )

    candidate_pool = [item for item in scored_candidates if int(item["score"]) > 0]
    if not candidate_pool:
        candidate_pool = [item for item in scored_candidates if int(item["score"]) == 0][:10]

    selected_candidates: list[dict[str, object]] = []
    for item in candidate_pool:
        term = str(item["term"])
        key = term.upper()
        if term.isupper():
            if any(
                len(str(existing["term"])) > len(term)
                and re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                    str(existing["term"]),
                    re.IGNORECASE,
                )
                for existing in selected_candidates
            ):
                continue
        selected_candidates.append(item)
        if len(selected_candidates) >= 40:
            break

    selected_terms = [str(item["term"]) for item in selected_candidates]
    dropped_terms = [str(item["term"]) for item in scored_candidates if item not in selected_candidates]
    priority_terms = [str(item["term"]) for item in scored_candidates]
    priority_trace = scored_candidates

    return {
        "dictionary_matches": dict_matches,
        "dictionary_occurrences": dict_occurrences,
        "lecture_title": lecture_title,
        "keyword_title_subjects": sorted(keyword_title_subjects),
        "title_subjects": sorted(title_subjects),
        "primary_title_subjects": sorted(primary_title_subjects),
        "secondary_title_subjects": sorted(secondary_title_subjects),
        "title_subjects_source": subject_router["source"],
        "subject_router": subject_router,
        "prefix_matches": prefix_matches,
        "suffix_matches": suffix_matches,
        "abbreviations": abbrev_matches,
        "allowlist": sorted(MEDICAL_MAP_ALLOWLIST),
        "hard_stopwords": sorted(MEDICAL_MAP_HARD_STOPWORDS),
        "prefix_occurrences": prefix_occurrences,
        "suffix_occurrences": suffix_occurrences,
        "abbreviation_occurrences": abbrev_occurrences,
        "priority_terms": priority_terms,
        "priority_trace": priority_trace,
        "scored_candidates": scored_candidates,
        "selected_terms": selected_terms,
        "dropped_terms": dropped_terms,
        "final_prompt": ", ".join(selected_terms),
    }


def build_injection_breakdown(text: str) -> dict:
    """
    extract_medical_keywords와 동일한 로직으로 카테고리별 키워드 분류 반환.
    injection 로그 기록에 사용.
    """
    return analyze_medical_keywords(text)


def create_run_log(audio_name: str) -> dict:
    """새 run 로그 딕셔너리 초기화."""
    return {
        "run_id":     f"{Path(audio_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "audio_file": audio_name,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "artifacts": {},
        "injection": {},
        "correction_layer": {
            "model":        "",
            "input_chars":  0,
            "output_chars": 0,
        },
        "highlight_extract_layer": {
            "model":        "",
            "input_chars":  0,
            "output_chars": 0,
            "chunks":       0,
        },
        "highlight_synthesis_layer": {
            "model":        "",
            "input_chars":  0,
            "output_chars": 0,
        },
        "format_layer": {
            "model":        "",
            "input_chars":  0,
            "output_chars": 0,
        },
    }


def group_audio_stem(stem: str | Path) -> str:
    """파일명 말미의 분할 suffix를 제거해 그룹 키를 만든다."""
    return GROUP_SUFFIX_RE.sub("", normalize_nfc(str(stem)))


def get_transcript_paths() -> list[Path]:
    paths = list(OUTPUT_DIR.glob("*/transcript.txt"))
    paths.extend(OUTPUT_DIR.glob("*_transcript.txt"))
    return sorted(set(paths))


def get_base_name_from_transcript_path(path: Path) -> str:
    if path.parent != OUTPUT_DIR and path.name == "transcript.txt":
        return path.parent.name
    return group_audio_stem(strip_known_suffix(path.stem, "_transcript"))


def get_corrected_transcript_paths() -> list[Path]:
    return sorted(OUTPUT_DIR.glob("*_corrected_transcript.txt"))


def get_base_name_from_corrected_path(path: Path) -> str:
    return group_audio_stem(strip_known_suffix(path.stem, "_corrected_transcript"))


def strip_known_suffix(stem: str, suffix: str) -> str:
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def collect_audio_files(file_arg: str | None = None) -> list[Path]:
    all_audio_files: list[Path] = []
    for ext in ["*.m4a", "*.mp3"]:
        all_audio_files.extend(INPUT_DIR.glob(ext))
    all_audio_files = sorted(all_audio_files)

    if not file_arg:
        return all_audio_files

    target = INPUT_DIR / file_arg
    if not target.exists():
        raise FileNotFoundError(target)

    target_base = group_audio_stem(target.stem)
    grouped = [path for path in all_audio_files if group_audio_stem(path.stem) == target_base]
    if not grouped:
        return [target]

    if len(grouped) > 1:
        grouped_names = ", ".join(path.name for path in grouped)
        print(f"🔗  관련 파트 자동 포함: {grouped_names}")

    return grouped


def save_run_log(run_log: dict) -> None:
    """run 로그를 logs/{run_id}.json 으로 저장."""
    run_path = LOG_DIR / f"{run_log['run_id']}.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)
    print(f"📊  Run 로그 저장: {run_path}")


def save_medical_map_debug(run_log: dict, matched_pdf: Path | None, debug_info: dict) -> None:
    """medical map 추출 디버그 정보를 별도 파일로 저장."""
    debug_path = LOG_DIR / f"{run_log['run_id']}_medical_map.json"
    payload = {
        "run_id": run_log["run_id"],
        "audio_file": run_log["audio_file"],
        "timestamp": run_log["timestamp"],
        "matched_pdf": matched_pdf.name if matched_pdf else None,
        "debug": debug_info,
    }
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"🧪  Medical Map 디버그 저장: {debug_path}")


# ════════════════════════════════════════════════════════════════════════════
#   오디오 전사
# ════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path: Path, medical_map: str = "",
                      lecture_title: str = "",
                      use_local_whisper: bool = False, model_size: str = "base",
                      compress_for_api: bool = False) -> str:
    """Whisper로 오디오 전사. medical_map을 initial_prompt로 주입."""
    
    # 25MB 제한 체크 (API 사용 시)
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    
    if not use_local_whisper and file_size_mb > 25:
        print(f"⚠️   파일 용량({file_size_mb:.1f}MB)이 OpenAI API 제한(25MB)을 초과합니다.")
        if compress_for_api:
            compressed_path = compress_audio_ffmpeg(audio_path, OUTPUT_DIR)
            if compressed_path is not None:
                compressed_mb = compressed_path.stat().st_size / (1024 * 1024)
                if compressed_mb <= 25:
                    print("✅  압축본이 25MB 이하이므로 단일 파일로 전사합니다.")
                    return transcribe_audio(
                        compressed_path,
                        medical_map,
                        lecture_title,
                        use_local_whisper=False,
                        model_size=model_size,
                        compress_for_api=False,
                    )
                print(f"⚠️   압축 후에도 {compressed_mb:.1f}MB로 제한 초과 → 분할 전사로 진행합니다.")
        splits = split_audio_ffmpeg(audio_path, OUTPUT_DIR)
        if splits:
            transcripts = []
            for i, part in enumerate(splits):
                print(f"🧩  조각 전사 중 ({i+1}/{len(splits)}): {part.name}")
                transcripts.append(transcribe_audio(
                    part,
                    medical_map,
                    lecture_title,
                    use_local_whisper=False,
                    model_size=model_size,
                    compress_for_api=False,
                ))
            return " ".join(transcripts)
        else:
            print("❌  분할 실패 → 전사를 진행할 수 없습니다. 파일을 수동으로 나누거나 faster-whisper를 설치해 주세요.")
            return ""

    print(f"🎙️  전사 중: '{audio_path.name}' ({'Local' if use_local_whisper else 'Remote API'})...")

    prompt_parts: list[str] = []
    if lecture_title:
        prompt_parts.append(f"강의명: {lecture_title}")
    if medical_map:
        prompt_parts.append(f"의학 용어: {medical_map}")
    elif MEDICAL_KEYWORDS:
        prompt_parts.append(MEDICAL_KEYWORDS)
    prompt = "\n".join(prompt_parts)

    if use_local_whisper:
        if WhisperModel is None:
            print("❌  faster-whisper 미설치 → OpenAI API fallback.")
            return transcribe_audio(audio_path, medical_map, lecture_title, use_local_whisper=False)
        print(f"⚙️  로컬 Whisper '{model_size}' 로드 중...")
        # faster-whisper/ctranslate2는 macOS에서 mps를 직접 지원하지 않으므로
        # 로컬 실행은 기본적으로 cpu/int8로 고정한다.
        device = "cpu"
        compute = "int8"
        print(f"🖥️   로컬 Whisper 런타임: device={device}, compute_type={compute}")
        model = WhisperModel(model_size, device=device, compute_type=compute)
        segments, _ = model.transcribe(str(audio_path), initial_prompt=prompt, beam_size=5)
        transcript = " ".join(seg.text for seg in segments)
        print(f"✅  로컬 전사 완료: '{audio_path.name}'")
        return transcript.strip()
    else:
        with open(audio_path, "rb") as f:
            transcription = whisper_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt=prompt,
                response_format="text"
            )
        print(f"✅  API 전사 완료: '{audio_path.name}'")
        return transcription


# ════════════════════════════════════════════════════════════════════════════
#   교정 레이어: 족보 기반 STT 오인식 교정
# ════════════════════════════════════════════════════════════════════════════

def _correction_user(medical_map: str, raw_transcript: str, lecture_title: str = "") -> str:
    terms = medical_map if medical_map else "제공된 용어 목록 없음"
    lecture_header = f"### 강의명\n{lecture_title}\n\n" if lecture_title else ""
    return (
        lecture_header +
        f"### 이번 강의 핵심 의학 용어 목록 (족보 추출)\n{terms}\n\n"
        f"### 원본 전사문\n{raw_transcript}\n\n"
        "위 전사문의 STT 오인식을 위 용어 목록을 참고하여 교정한 전사문만 출력하세요."
    )


def correct_with_jokbo(
    raw_transcript: str,
    medical_map: str,
    lecture_title: str = "",
    use_local: bool = False,
    use_openrouter: bool = False,
    model_id: str | None = None,
    use_reasoning: bool = False,
    run_log: dict | None = None,
) -> str:
    """
    족보를 레퍼런스로 삼아 전사문의 STT 오인식을 LLM으로 교정.
    실패 시 원본 전사문 반환.
    """
    if use_local:
        client = OLLAMA_CLIENT
        default_model = "qwen2.5:7b"
    elif use_openrouter:
        client = openrouter_client
        default_model = DEFAULT_OPENROUTER_MODEL
    else:
        client = hf_client
        default_model = RELIABLE_HF_MODEL_ID

    if client is None:
        print("  ❌  교정 레이어: LLM 클라이언트 미초기화 → 원본 반환")
        return raw_transcript

    target_model = model_id if model_id else default_model
    print(f"✏️   교정 레이어: {target_model} 으로 오인식 교정 중...")

    if run_log is not None:
        run_log["correction_layer"]["input_chars"] = len(raw_transcript)
        run_log["correction_layer"]["model"] = target_model

    system_msg = CORRECTION_SYSTEM
    user_msg   = _correction_user(medical_map, raw_transcript, lecture_title=lecture_title)
    max_tokens = _safe_max_tokens(system_msg, user_msg)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.05,
            }
            if use_openrouter and use_reasoning:
                kwargs["extra_body"] = {"reasoning": {"enabled": True}}

            resp = client.chat.completions.create(**kwargs)
            if not resp or not resp.choices:
                raise ValueError("API 응답이 비어있거나 올바르지 않습니다.")
            
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("API 응답 내용(content)이 None입니다.")

            corrected = content.strip()
            if run_log is not None:
                run_log["correction_layer"]["output_chars"] = len(corrected)
            print(f"  ✅  교정 완료: {len(raw_transcript):,}자 → {len(corrected):,}자")
            return corrected
        except Exception as e:
            if not use_local and ("504" in str(e) or "timeout" in str(e).lower()):
                print(f"  ⚠️  Timeout. 재시도 ({attempt + 1}/{max_retries})...")
                time.sleep(2)
                continue
            print(f"  ❌  교정 레이어 오류: {e} → 원본 반환")
            return raw_transcript

    print("  ❌  교정 레이어 재시도 소진 → 원본 반환")
    return raw_transcript


# ════════════════════════════════════════════════════════════════════════════
#   하이라이트 레이어: 청크 추출 + 상위 통합
# ════════════════════════════════════════════════════════════════════════════

def get_llm_client_and_default_model(
    *,
    use_local: bool = False,
    use_openrouter: bool = False,
    purpose: str = "correction",
) -> tuple[OpenAI | None, str]:
    if use_local:
        return OLLAMA_CLIENT, "qwen2.5:7b"
    if use_openrouter:
        if purpose == "highlight_extract":
            return openrouter_client, DEFAULT_HIGHLIGHT_EXTRACT_MODEL
        if purpose == "highlight_synthesis":
            return openrouter_client, DEFAULT_HIGHLIGHT_SYNTHESIS_MODEL
        return openrouter_client, DEFAULT_OPENROUTER_MODEL
    return hf_client, RELIABLE_HF_MODEL_ID


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 1.1))


def chunk_text_by_estimated_tokens(
    text: str,
    *,
    chunk_tokens: int = 2200,
    overlap_tokens: int = 250,
) -> list[dict[str, object]]:
    if not text.strip():
        return []

    chunk_chars = int(chunk_tokens * 1.1)
    overlap_chars = int(overlap_tokens * 1.1)
    min_search_start = int(chunk_chars * 0.65)
    chunks: list[dict[str, object]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        provisional_end = min(start + chunk_chars, text_len)
        if provisional_end >= text_len:
            end = text_len
        else:
            search_region = text[start + min_search_start:provisional_end]
            boundary = max(
                search_region.rfind("\n"),
                search_region.rfind(". "),
                search_region.rfind("? "),
                search_region.rfind("! "),
                search_region.rfind(".\n"),
            )
            end = provisional_end if boundary < 0 else start + min_search_start + boundary + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "start_char": start,
                "end_char": end,
                "estimated_tokens": estimate_tokens(chunk_text),
                "text": chunk_text,
            })

        if end >= text_len:
            break
        start = max(end - overlap_chars, start + 1)

    return chunks


def save_chunk_highlights(run_log: dict, chunk_records: list[dict]) -> Path:
    path = LOG_DIR / f"{run_log['run_id']}_chunk_highlights.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for record in chunk_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"🧩  Chunk highlights 저장: {path}", flush=True)
    return path


def init_chunk_highlights_file(run_log: dict) -> Path:
    path = LOG_DIR / f"{run_log['run_id']}_chunk_highlights.jsonl"
    path.write_text("", encoding="utf-8")
    return path


def append_chunk_highlight(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def render_extracted_notes(chunk_records: list[dict], *, lecture_title: str = "") -> str:
    lines: list[str] = []
    if lecture_title:
        lines.append(f"# {lecture_title}")
        lines.append("")
    lines.append("# Chunk Extracted Notes")
    lines.append("")
    for record in chunk_records:
        extracted = str(record.get("extracted_text", "")).strip()
        if not extracted:
            continue
        lines.append(f"## Chunk {record.get('chunk_id')}")
        lines.append(extracted)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_extracted_notes_for_prompt(chunk_records: list[dict]) -> str:
    lines: list[str] = []
    for record in chunk_records:
        extracted = str(record.get("extracted_text", "")).strip()
        if not extracted:
            continue
        lines.append(f"## Chunk {record.get('chunk_id')}")
        lines.append(extracted)
        lines.append("")
    return "\n".join(lines).strip()


def _highlight_extract_system() -> str:
    return (
        "당신은 의학 강의 청크 추출기입니다.\n"
        "요약이 아니라 추출만 하세요.\n"
        "새 의학 사실을 만들지 말고, 청크에 있는 내용만 더 읽기 쉬운 문장으로 정리하세요.\n"
        "불확실하면 과감히 생략하고, 확실한 정보만 남기세요.\n"
        "JSON, 코드펜스, 번호 목록 없이 평문/마크다운만 출력하세요."
    )


def _highlight_extract_user(
    chunk: dict[str, object],
    *,
    lecture_title: str = "",
    subject_hint: str = "",
) -> str:
    chunk_text = str(chunk["text"])
    return (
        f"강의명: {lecture_title or '미지정'}\n"
        f"큰 분과: {subject_hint or '미지정'}\n"
        f"청크 번호: {chunk['chunk_id']}\n"
        "작업:\n"
        "- 첫 줄은 짧은 소제목 1개로 쓰세요.\n"
        "- 그 아래에 중요한 의학 내용만 4~5줄로 다시 적으세요.\n"
        "- 문장은 짧고 단정하게 쓰세요.\n"
        "- 하위 레이어이므로 자세한 설명, 중복, 장황한 문장은 피하세요.\n"
        "- 확실하지 않은 말은 억지로 정리하지 말고 빼세요.\n"
        "- bullet 없이 일반 줄글/줄바꿈 형식으로 쓰세요.\n"
        "- 예시 형식:\n"
        "미숙아 수분 및 수액 관리 원칙\n"
        "34주부터 안전한 경우 경구 섭취를 시도한다.\n"
        "피부 미성숙과 넓은 체표면적으로 insensible water loss가 많다.\n"
        "체중 증가, 소변/대변량을 비교해 수분 용량을 조절한다.\n\n"
        f"청크 원문:\n{chunk_text}"
    )


def _highlight_synthesis_system() -> str:
    return (
        "당신은 의학 강의 하이라이트 통합기입니다.\n"
        "청크 추출 결과를 병합해 중복을 줄이되, 중요한 세부 내용은 보존하세요.\n"
        "너무 짧게 압축하지 말고, 강의의 범위와 디테일이 충분히 남도록 정리하세요.\n"
        "새 사실을 만들지 말고 제공된 추출 결과만 바탕으로 정리하세요.\n"
        "마크다운만 출력하세요."
    )


def _highlight_synthesis_user(
    chunk_records: list[dict],
    *,
    lecture_title: str = "",
    subject_hint: str = "",
) -> str:
    extracted_notes = _render_extracted_notes_for_prompt(chunk_records)
    return (
        f"강의명: {lecture_title or '미지정'}\n"
        f"큰 분과: {subject_hint or '미지정'}\n\n"
        "아래는 corrected transcript를 청크 단위로 정제한 추출 노트입니다.\n"
        "이걸 바탕으로 최종 정리본을 만드세요.\n\n"
        "출력 형식:\n"
        "## 핵심 개념\n"
        "- 8개 이상, 누락보다 보존을 우선\n"
        "## 시험 포인트\n"
        "- 8개 이상, 수치/감별/처치 원칙 포함\n"
        "## 구조화 메모\n"
        "- 주제별 소제목과 bullet로 충분히 자세히\n"
        "## 재청취 필요 구간\n"
        "- ...\n\n"
        "빈 청크는 무시하세요.\n"
        "새 사실을 만들지 말고, 추출 노트에 있는 내용만 재조직화하세요.\n"
        "의미가 다른 포인트를 하나로 과도하게 합치지 마세요.\n"
        "가능하면 고유 질환명, 검사명, 감별 포인트, 수치 기준을 보존하세요.\n"
        "최종 결과가 지나치게 짧아지지 않게 하세요.\n"
        "압축보다 보존을 우선하세요.\n\n"
        f"청크 추출 노트:\n{extracted_notes}"
    )


def extract_chunk_highlights(
    corrected_transcript: str,
    subject_hint: str = "",
    lecture_title: str = "",
    *,
    use_local: bool = False,
    use_openrouter: bool = False,
    model_id: str | None = None,
    chunk_tokens: int = 1500,
    overlap_tokens: int = 100,
    run_log: dict | None = None,
) -> list[dict]:
    client, default_model = get_llm_client_and_default_model(
        use_local=use_local,
        use_openrouter=use_openrouter,
        purpose="highlight_extract",
    )
    if client is None:
        print("  ❌  하이라이트 추출 레이어: LLM 클라이언트 미초기화")
        return []

    target_model = model_id or default_model
    chunks = chunk_text_by_estimated_tokens(
        corrected_transcript,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    print(f"🧩  하이라이트 추출: {target_model} / {len(chunks)}개 청크", flush=True)

    if run_log is not None:
        run_log["highlight_extract_layer"]["model"] = target_model
        run_log["highlight_extract_layer"]["input_chars"] = len(corrected_transcript)
        run_log["highlight_extract_layer"]["chunks"] = len(chunks)
        chunk_log_path = init_chunk_highlights_file(run_log)
        run_log["artifacts"]["chunk_highlights"] = str(chunk_log_path)
    else:
        chunk_log_path = None

    results: list[dict] = []
    for chunk in chunks:
        system_msg = _highlight_extract_system()
        user_msg = _highlight_extract_user(chunk, lecture_title=lecture_title, subject_hint=subject_hint)
        max_tokens = HIGHLIGHT_EXTRACT_MAX_TOKENS
        print(f"  📦  청크 추출 중 ({chunk['chunk_id']}/{len(chunks)})...", flush=True)

        extracted_text = ""
        for attempt in range(3):
            try:
                kwargs = {
                    "model": target_model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                }
                if use_openrouter:
                    kwargs["extra_body"] = {"reasoning": {"effort": "none", "exclude": True}}
                resp = client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                content = re.sub(r"^```(?:markdown|md|text)?\s*", "", content, flags=re.IGNORECASE)
                content = re.sub(r"\s*```$", "", content)
                if content:
                    extracted_text = content
                    break
            except Exception as e:
                print(f"    ⚠️  청크 {chunk['chunk_id']} 시도 {attempt+1} 실패: {e}", flush=True)
                time.sleep(1)

        record = {
            "chunk_id": int(chunk["chunk_id"]),
            "start_char": int(chunk["start_char"]),
            "end_char": int(chunk["end_char"]),
            "estimated_tokens": int(chunk["estimated_tokens"]),
            "extracted_text": extracted_text,
        }
        results.append(record)
        if chunk_log_path is not None:
            append_chunk_highlight(chunk_log_path, record)
        status = f"{len(extracted_text)}자" if extracted_text else "빈 추출"
        print(f"    ✅  청크 {chunk['chunk_id']} 완료 ({status})", flush=True)

    if run_log is not None:
        run_log["highlight_extract_layer"]["output_chars"] = sum(
            len(json.dumps(item, ensure_ascii=False)) for item in results
        )
    return results


def synthesize_study_brief(
    chunk_records: list[dict],
    subject_hint: str = "",
    lecture_title: str = "",
    *,
    use_local: bool = False,
    use_openrouter: bool = False,
    model_id: str | None = None,
    run_log: dict | None = None,
) -> str | None:
    client, default_model = get_llm_client_and_default_model(
        use_local=use_local,
        use_openrouter=use_openrouter,
        purpose="highlight_synthesis",
    )
    if client is None:
        print("  ❌  하이라이트 통합 레이어: LLM 클라이언트 미초기화")
        return None

    target_model = model_id or default_model
    system_msg = _highlight_synthesis_system()
    user_msg = _highlight_synthesis_user(chunk_records, lecture_title=lecture_title, subject_hint=subject_hint)
    max_tokens = _bounded_max_tokens(
        system_msg,
        user_msg,
        desired=HIGHLIGHT_SYNTHESIS_MAX_TOKENS,
        floor=1200,
    )
    print(f"🧠  하이라이트 통합: {target_model} / max_tokens={max_tokens}", flush=True)

    if run_log is not None:
        run_log["highlight_synthesis_layer"]["model"] = target_model
        run_log["highlight_synthesis_layer"]["input_chars"] = len(user_msg)

    for attempt in range(3):
        try:
            kwargs = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }
            if use_openrouter:
                kwargs["extra_body"] = {"reasoning": {"effort": "none", "exclude": True}}
            resp = client.chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            if content:
                if run_log is not None:
                    run_log["highlight_synthesis_layer"]["output_chars"] = len(content)
                return content
        except Exception as e:
            print(f"  ⚠️  하이라이트 통합 시도 {attempt+1} 실패: {e}", flush=True)
            time.sleep(1)

    print("  ❌  하이라이트 통합 실패", flush=True)
    return None


# ════════════════════════════════════════════════════════════════════════════
#   Postprocessing 오케스트레이터
# ════════════════════════════════════════════════════════════════════════════

def postprocess_transcript(
    raw_transcript: str,
    jokbo_text: str,
    medical_map: str,
    lecture_title: str = "",
    highlight_subject_hint: str = "",
    use_local: bool = False,
    use_openrouter: bool = False,
    model_id: str | None = None,
    use_reasoning: bool = False,
    run_log: dict | None = None,
    correction_only: bool = False,
    generate_study_brief: bool = False,
    highlight_extract_model: str | None = None,
    highlight_synthesis_model: str | None = None,
    highlight_chunk_tokens: int = 1500,
    highlight_overlap_tokens: int = 100,
) -> tuple[str, str | None, list[dict]]:
    """
    3-step postprocessing:
    Step 1 → correct_with_jokbo
    Step 2 → extract_chunk_highlights
    Step 3 → synthesize_study_brief
    """
    print("✏️   Step 1: 족보 기반 오인식 교정 중...")
    corrected = correct_with_jokbo(
        raw_transcript, medical_map, lecture_title=lecture_title,
        use_local=use_local, use_openrouter=use_openrouter,
        model_id=model_id, use_reasoning=use_reasoning, run_log=run_log
    )

    if correction_only:
        print("📝  correction-only 모드: 최종 하이라이트 레이어를 건너뜁니다.")
        return corrected, None, []

    print("📝  Step 2: 청크별 강조점 추출 중...")
    chunk_records = extract_chunk_highlights(
        corrected, highlight_subject_hint, lecture_title=lecture_title,
        use_local=use_local, use_openrouter=use_openrouter,
        model_id=highlight_extract_model,
        chunk_tokens=highlight_chunk_tokens,
        overlap_tokens=highlight_overlap_tokens,
        run_log=run_log,
    )
    if not generate_study_brief:
        print("📝  extracted-notes 기본 모드: study brief 생성을 건너뜁니다.")
        return corrected, None, chunk_records

    print("📝  Step 3: 최종 강조점 통합 중...")
    notes = synthesize_study_brief(
        chunk_records, highlight_subject_hint, lecture_title=lecture_title,
        use_local=use_local, use_openrouter=use_openrouter,
        model_id=highlight_synthesis_model,
        run_log=run_log,
    )
    return corrected, notes, chunk_records


# ════════════════════════════════════════════════════════════════════════════
#   main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Medical Transcription and Summarization")
    parser.add_argument("--whisper-only",   action="store_true",
                        help="Whisper 전사만 실행 (LLM postprocessing 생략)")
    parser.add_argument("--correction-only", action="store_true",
                        help="Whisper 전사 후 교정본까지만 저장 (하이라이트 레이어 생략)")
    parser.add_argument("--highlights-only", action="store_true",
                        help="기존 corrected transcript로 하이라이트 레이어만 실행")
    parser.add_argument("--debug-medical-map", action="store_true",
                        help="medical map 추출 디버그 정보를 logs/ 에 별도 저장")
    parser.add_argument("--compress-for-api", action="store_true",
                        help="25MB 초과 시 Whisper API 업로드 전 경량 압축을 먼저 시도")
    parser.add_argument("--local-whisper",  action="store_true",
                        help="로컬 faster-whisper 사용 (OpenAI API 대신)")
    parser.add_argument("--whisper-model",  type=str, default="base",
                        help="로컬 Whisper 모델 크기 (base/small/medium/large-v3)")
    parser.add_argument("--file",           type=str, default=None,
                        help="처리할 특정 오디오 파일명 (input/ 내, 예: '건사 테스트용.m4a')")
    parser.add_argument("--lecture-title",  type=str, default=None,
                        help="강의명 수동 지정 (medical map/프롬프트 boost용)")
    parser.add_argument("--disable-subject-router", action="store_true",
                        help="강의명 subject 라우터 비활성화 (키워드 fallback만 사용)")
    parser.add_argument("--subject-router-model", type=str, default=DEFAULT_SUBJECT_ROUTER_MODEL,
                        help=f"강의명 subject 라우터 모델 (기본: {DEFAULT_SUBJECT_ROUTER_MODEL})")
    parser.add_argument("--reprocess",      action="store_true",
                        help="output/ 기존 transcript를 재처리")
    parser.add_argument("--highlight-extract-model", type=str, default=DEFAULT_HIGHLIGHT_EXTRACT_MODEL,
                        help=f"청크 추출 모델 (기본: {DEFAULT_HIGHLIGHT_EXTRACT_MODEL})")
    parser.add_argument("--study-brief", action="store_true",
                        help="최종 study brief까지 추가 생성 (기본은 extracted notes만 저장)")
    parser.add_argument("--highlight-synthesis-model", type=str, default=DEFAULT_HIGHLIGHT_SYNTHESIS_MODEL,
                        help=f"최종 통합 모델 (기본: {DEFAULT_HIGHLIGHT_SYNTHESIS_MODEL})")
    parser.add_argument("--highlight-chunk-tokens", type=int, default=1500,
                        help="청크 추출용 목표 토큰 수 (기본: 1500)")
    parser.add_argument("--highlight-overlap-tokens", type=int, default=100,
                        help="청크 겹침 토큰 수 (기본: 100)")
    parser.add_argument("--local",          action="store_true",
                        help="로컬 Ollama 사용 (HF Router 대신)")
    parser.add_argument("--openrouter",     action="store_true",
                        help="OpenRouter API 사용")
    parser.add_argument("--model",          type=str,
                        help="LLM 모델 ID 직접 지정")
    parser.add_argument("--context-limit",  type=int, default=200000,
                        help="LLM 컨텍스트 창 크기 (기본: 200000)")
    parser.add_argument("--use-reasoning",  action="store_true",
                        help="OpenRouter에서 추론(Reasoning) 모드 활성화")
    args = parser.parse_args()

    global CONTEXT_LIMIT
    CONTEXT_LIMIT = args.context_limit
    if args.context_limit != 200000:
        print(f"📐  컨텍스트 한계: {CONTEXT_LIMIT:,} 토큰")

    if not args.reprocess and not args.local_whisper and not whisper_client:
        print("❌  OPENAI_API_KEY 미설정 (Remote Whisper API 필요)")
        return
    if not args.whisper_only and not args.local and not args.openrouter and not hf_client:
        print("❌  HF_API_KEY 미설정 (Remote LLM 필요)")
        return
    if args.openrouter and not openrouter_client:
        print("❌  OPENROUTER_API_KEY 미설정 (OpenRouter 필요)")
        return

    print("🚀  Medical Lecture Processor 시작...")
    if args.whisper_only:
        print("💡  모드: Whisper 전사만 (LLM 생략)")
    elif args.correction_only:
        print("💡  모드: 교정본까지만 저장 (포맷 레이어 생략)")
    elif args.highlights_only:
        print("💡  모드: corrected transcript -> 하이라이트 레이어만 실행")
    else:
        print("💡  모드: 교정본 + 최종 강조점 정리본 생성")

    enabled_modes = sum(bool(x) for x in [args.whisper_only, args.correction_only, args.highlights_only])
    if enabled_modes > 1:
        print("❌  --whisper-only, --correction-only, --highlights-only 는 함께 사용할 수 없습니다.")
        return

    # ── --reprocess 모드 ────────────────────────────────────────────────────
    if args.reprocess:
        print("🔄  기존 transcript 재처리 중...")
        transcript_files = get_transcript_paths()
        if args.file:
            target_base = group_audio_stem(Path(args.file).stem)
            transcript_files = [
                tp for tp in transcript_files
                if get_base_name_from_transcript_path(tp) == target_base
            ]
            if transcript_files:
                print(f"🎯  재처리 대상 제한: {', '.join(tp.name for tp in sorted(transcript_files))}")
        if not transcript_files:
            print(f"⚠️   '{OUTPUT_DIR}'에 transcript 파일 없음.")
            return

        for tp in transcript_files:
            base_name = get_base_name_from_transcript_path(tp)
            print(f"\n--- 재처리: {base_name} ---")
            with open(tp, "r", encoding="utf-8") as f:
                transcript = f.read()

            # ── 해당 파일에 맞는 족보 매칭 ──────────────────────────────────
            jokbo_text = ""
            medical_map = ""
            medical_map_debug = {}
            lecture_title = args.lecture_title or ""
            matched_pdf = find_matching_jokbo(base_name, JOKBO_DIR)
            if matched_pdf:
                lecture_title = lecture_title or extract_lecture_title(matched_pdf.name)
                jokbo_text = extract_jukbo_section(matched_pdf)
                medical_map_debug = analyze_medical_keywords(
                    jokbo_text,
                    lecture_title=lecture_title,
                    subject_router_model=args.subject_router_model,
                    use_subject_router=not args.disable_subject_router,
                )
                medical_map = medical_map_debug["final_prompt"]
            else:
                print(f"⚠️   [경고] '{base_name}'에 매칭되는 족보를 찾지 못했습니다.")
                # reprocess에서는 인터랙티브 대기보다 경고 후 진행 (혹은 필요시 추가)

            run_log = create_run_log(base_name)
            run_log["injection"] = {
                **medical_map_debug,
                "lecture_title": lecture_title,
                "jokbo_section_chars": len(jokbo_text),
                "final_prompt": medical_map,
            }
            if args.debug_medical_map and jokbo_text:
                save_medical_map_debug(run_log, matched_pdf, run_log["injection"])

            highlight_subjects = set(medical_map_debug.get("title_subjects", [])) or infer_title_subjects(lecture_title)
            highlight_subject_hint = build_highlight_subject_hint(highlight_subjects)

            corrected, notes, chunk_records = postprocess_transcript(
                transcript, jokbo_text, medical_map, lecture_title=lecture_title,
                highlight_subject_hint=highlight_subject_hint,
                use_local=args.local, use_openrouter=args.openrouter,
                model_id=args.model, use_reasoning=args.use_reasoning,
                run_log=run_log,
                correction_only=args.correction_only,
                generate_study_brief=args.study_brief,
                highlight_extract_model=args.highlight_extract_model,
                highlight_synthesis_model=args.highlight_synthesis_model,
                highlight_chunk_tokens=args.highlight_chunk_tokens,
                highlight_overlap_tokens=args.highlight_overlap_tokens,
            )

            corrected_out = OUTPUT_DIR / f"{base_name}_corrected_transcript.txt"
            with open(corrected_out, "w", encoding="utf-8") as f:
                f.write(corrected)
            print(f"🎉  corrected transcript 저장: {corrected_out}")
            run_log["artifacts"]["corrected_transcript"] = str(corrected_out)

            if chunk_records:
                chunk_path = save_chunk_highlights(run_log, chunk_records)
                run_log["artifacts"]["chunk_highlights"] = str(chunk_path)
                extracted_out = OUTPUT_DIR / f"{base_name}_extracted_notes.md"
                extracted_md = render_extracted_notes(chunk_records, lecture_title=lecture_title or base_name)
                with open(extracted_out, "w", encoding="utf-8") as f:
                    f.write(extracted_md)
                print(f"🎉  extracted notes 저장: {extracted_out}")
                run_log["artifacts"]["extracted_notes"] = str(extracted_out)

            if notes:
                out = OUTPUT_DIR / f"{base_name}_study_brief.md"
                with open(out, "w", encoding="utf-8") as f:
                    f.write(notes)
                print(f"🎉  study brief 저장: {out}")
                run_log["artifacts"]["study_brief"] = str(out)

            save_run_log(run_log)
        return

    # ── --highlights-only 모드 ─────────────────────────────────────────────
    if args.highlights_only:
        print("🔄  기존 corrected transcript로 하이라이트 생성 중...")
        corrected_files = get_corrected_transcript_paths()
        if args.file:
            target_base = group_audio_stem(Path(args.file).stem)
            corrected_files = [
                cp for cp in corrected_files
                if get_base_name_from_corrected_path(cp) == target_base
            ]
            if corrected_files:
                print(f"🎯  하이라이트 대상 제한: {', '.join(cp.name for cp in sorted(corrected_files))}")
        if not corrected_files:
            print(f"⚠️   '{OUTPUT_DIR}'에 corrected transcript 파일 없음.")
            return

        for cp in corrected_files:
            base_name = get_base_name_from_corrected_path(cp)
            print(f"\n--- 하이라이트 생성: {base_name} ---")
            with open(cp, "r", encoding="utf-8") as f:
                corrected_transcript = f.read()

            jokbo_text = ""
            medical_map = ""
            medical_map_debug = {}
            lecture_title = args.lecture_title or ""
            matched_pdf = find_matching_jokbo(base_name, JOKBO_DIR)
            if matched_pdf:
                lecture_title = lecture_title or extract_lecture_title(matched_pdf.name)
                jokbo_text = extract_jukbo_section(matched_pdf)
                medical_map_debug = analyze_medical_keywords(
                    jokbo_text,
                    lecture_title=lecture_title,
                    subject_router_model=args.subject_router_model,
                    use_subject_router=not args.disable_subject_router,
                )
                medical_map = medical_map_debug["final_prompt"]
            else:
                print(f"⚠️   [경고] '{base_name}'에 매칭되는 족보를 찾지 못했습니다.")

            run_log = create_run_log(f"{base_name}_highlights")
            run_log["injection"] = {
                **medical_map_debug,
                "lecture_title": lecture_title,
                "jokbo_section_chars": len(jokbo_text),
                "final_prompt": medical_map,
            }
            if args.debug_medical_map and jokbo_text:
                save_medical_map_debug(run_log, matched_pdf, run_log["injection"])

            highlight_subjects = set(medical_map_debug.get("title_subjects", [])) or infer_title_subjects(lecture_title)
            highlight_subject_hint = build_highlight_subject_hint(highlight_subjects)

            chunk_records = extract_chunk_highlights(
                corrected_transcript,
                highlight_subject_hint,
                lecture_title=lecture_title,
                use_local=args.local,
                use_openrouter=args.openrouter,
                model_id=args.highlight_extract_model,
                chunk_tokens=args.highlight_chunk_tokens,
                overlap_tokens=args.highlight_overlap_tokens,
                run_log=run_log,
            )
            chunk_path = save_chunk_highlights(run_log, chunk_records)
            run_log["artifacts"]["chunk_highlights"] = str(chunk_path)
            extracted_out = OUTPUT_DIR / f"{base_name}_extracted_notes.md"
            extracted_md = render_extracted_notes(chunk_records, lecture_title=lecture_title or base_name)
            with open(extracted_out, "w", encoding="utf-8") as f:
                f.write(extracted_md)
            print(f"🎉  Extracted notes 저장: {extracted_out}")
            run_log["artifacts"]["extracted_notes"] = str(extracted_out)

            if args.study_brief:
                notes = synthesize_study_brief(
                    chunk_records,
                    highlight_subject_hint,
                    lecture_title=lecture_title,
                    use_local=args.local,
                    use_openrouter=args.openrouter,
                    model_id=args.highlight_synthesis_model,
                    run_log=run_log,
                )
                if notes:
                    out = OUTPUT_DIR / f"{base_name}_study_brief.md"
                    with open(out, "w", encoding="utf-8") as f:
                        f.write(notes)
                    print(f"🎉  Study brief 저장: {out}")
                    run_log["artifacts"]["study_brief"] = str(out)
            else:
                print("📝  extracted-notes 기본 모드: study brief 생성을 건너뜁니다.")

            save_run_log(run_log)
        return

    # ── 오디오 파일 처리 ─────────────────────────────────────────────────────
    try:
        audio_files = collect_audio_files(args.file)
    except FileNotFoundError as exc:
        print(f"❌  파일 없음: {exc.args[0]}")
        return

    if not audio_files:
        print(f"⚠️   '{INPUT_DIR}'에 오디오 파일(.m4a/.mp3) 없음.")
        return

    # 논리적 그룹화: 파일명_1, 파일명_2 등을 하나의 그룹으로 묶음
    grouped_files = {}
    for path in sorted(audio_files):
        # '_1', '_2' 또는 ' (1)', ' (2)' 등의 패턴 제거하여 베이스 이름 추출
        base_name = group_audio_stem(path.stem)
        if base_name not in grouped_files:
            grouped_files[base_name] = []
        grouped_files[base_name].append(path)

    for base_name, group in grouped_files.items():
        print(f"\n--- 그룹 처리 중: {base_name} ({len(group)}개 파일) ---")

        # ── 해당 그룹에 맞는 족보 매칭 (첫 번째 파일 기준) ──────────────────
        jokbo_text = ""
        medical_map = ""
        medical_map_debug = {}
        lecture_title = args.lecture_title or ""
        
        # 그룹의 첫 번째 파일로 매칭 시도
        representative_file = group[0]
        matched_pdf = find_matching_jokbo(representative_file.name, JOKBO_DIR)
        
        if matched_pdf:
            lecture_title = lecture_title or extract_lecture_title(matched_pdf.name)
            jokbo_text = extract_jukbo_section(matched_pdf)
        else:
            a_date, a_session, a_prof = extract_match_keys(representative_file.name)
            lecture_title = lecture_title or extract_lecture_title(representative_file.name)
            print(f"⚠️   [경고] '{base_name}'에 매칭되는 족보를 찾지 못했습니다. (날짜: {a_date}, 교시: {a_session}, 교수: {a_prof})")
            choice = input("    족보 없이 진행할까요? (y/n): ").strip().lower()
            if choice != 'y':
                print(f"🛑  그룹을 스킵합니다: {base_name}")
                continue
            print("    ⏩  족보 없이 진행합니다.")

        if jokbo_text:
            medical_map_debug = analyze_medical_keywords(
                jokbo_text,
                lecture_title=lecture_title,
                subject_router_model=args.subject_router_model,
                use_subject_router=not args.disable_subject_router,
            )
            medical_map = medical_map_debug["final_prompt"]
            kw_count = len(medical_map.split(","))
            print(f"🗺️   Medical Map: {kw_count}개 키워드 생성.")
            if lecture_title:
                print(f"🏷️   강의명 prior: {lecture_title}")

        # 1. 그룹 내 모든 파일 전사 및 합치기
        all_transcripts = []
        for audio_path in group:
            print(f"📂  파이너 전사: {audio_path.name}")
            transcript = transcribe_audio(
                audio_path,
                medical_map=medical_map,
                lecture_title=lecture_title,
                use_local_whisper=args.local_whisper,
                model_size=args.whisper_model,
                compress_for_api=args.compress_for_api,
            )
            all_transcripts.append(transcript)

        combined_transcript = "\n\n".join(all_transcripts)

        # run 로그 초기화 (그룹 단위)
        run_log = create_run_log(base_name)
        run_log["injection"] = {
            **medical_map_debug,
            "lecture_title": lecture_title,
            "jokbo_section_chars": len(jokbo_text),
            "final_prompt": medical_map,
        }
        if args.debug_medical_map and jokbo_text:
            save_medical_map_debug(run_log, matched_pdf, run_log["injection"])

        highlight_subjects = set(medical_map_debug.get("title_subjects", [])) or infer_title_subjects(lecture_title)
        highlight_subject_hint = build_highlight_subject_hint(highlight_subjects)

        # transcript 저장
        tp = OUTPUT_DIR / f"{base_name}_transcript.txt"
        with open(tp, "w", encoding="utf-8") as f:
            f.write(combined_transcript)
        print(f"💾  통합 Transcript 저장: {tp}")
        run_log["artifacts"]["transcript"] = str(tp)

        if args.whisper_only:
            save_run_log(run_log)
            continue

        corrected, notes, chunk_records = postprocess_transcript(
            combined_transcript, jokbo_text, medical_map, lecture_title=lecture_title,
            highlight_subject_hint=highlight_subject_hint,
            use_local=args.local, use_openrouter=args.openrouter,
            model_id=args.model, use_reasoning=args.use_reasoning,
            run_log=run_log,
            correction_only=args.correction_only,
            generate_study_brief=args.study_brief,
            highlight_extract_model=args.highlight_extract_model,
            highlight_synthesis_model=args.highlight_synthesis_model,
            highlight_chunk_tokens=args.highlight_chunk_tokens,
            highlight_overlap_tokens=args.highlight_overlap_tokens,
        )

        corrected_path = OUTPUT_DIR / f"{base_name}_corrected_transcript.txt"
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(corrected)
        print(f"🎉  Corrected transcript 저장: {corrected_path}")
        run_log["artifacts"]["corrected_transcript"] = str(corrected_path)

        if chunk_records:
            chunk_path = save_chunk_highlights(run_log, chunk_records)
            run_log["artifacts"]["chunk_highlights"] = str(chunk_path)
            extracted_out = OUTPUT_DIR / f"{base_name}_extracted_notes.md"
            extracted_md = render_extracted_notes(chunk_records, lecture_title=lecture_title or base_name)
            with open(extracted_out, "w", encoding="utf-8") as f:
                f.write(extracted_md)
            print(f"🎉  Extracted notes 저장: {extracted_out}")
            run_log["artifacts"]["extracted_notes"] = str(extracted_out)

        if notes:
            out = OUTPUT_DIR / f"{base_name}_study_brief.md"
            with open(out, "w", encoding="utf-8") as f:
                f.write(notes)
            print(f"🎉  Study brief 저장: {out}")
            run_log["artifacts"]["study_brief"] = str(out)

        save_run_log(run_log)

    print("\n🏁  전체 처리 완료!")


if __name__ == "__main__":
    main()
