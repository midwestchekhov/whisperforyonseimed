"""
Whisper injection 키워드 추출 정규식 테스트 스크립트.
extract_medical_keywords() 와 동일한 로직을 단계별로 분해해서 확인.
"""
import re
import unicodedata
from collections import Counter

# ── 수정된 main.py regex 동기화 ───────────────────────────────────────────────
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
# 순수 대문자(ECG, MI, SGLT2) + 소문자 혼용(HbA1c, eGFR, pH, mRNA)
ABBREV_RE = re.compile(
    r'(?<![A-Za-z])'
    r'(?:[A-Z]{2,}[0-9]*|[A-Za-z][a-z0-9]*[A-Z][A-Za-z0-9]*)'
    r'(?![A-Za-z0-9])'
)

KOR_STOPWORDS = {
    "있는", "없는", "것이", "때문", "에서", "으로", "에게", "그리고",
    "하지만", "그러나", "따라서", "그래서", "이것", "저것", "이런",
    "저런", "어떤", "모든", "각각", "이때", "경우", "사실", "내용",
    "부분", "정도", "수가", "있다", "없다", "된다", "하는", "이다",
}

# ── 샘플 텍스트 (족보/강의 내용 시뮬레이션) ──────────────────────────────────
SAMPLE_TEXT = """
Hypertension과 hyperglycemia는 당뇨병 환자에서 흔히 동반됩니다.
심근경색(MI) 환자에서 ECG 소견을 확인하고 STEMI vs NSTEMI를 감별해야 합니다.
Pneumonia의 치료에서 antibiotic 선택이 중요하며, polymyxin은 내성균에 사용합니다.
심방세동(AF)과 DVT에서 anticoagulation 요법을 시작합니다.
hepatitis로 인한 jaundice와 encephalopathy를 주의해야 합니다.
Lymphoma 환자에서 chemotherapy 부작용으로 neutropenia가 발생합니다.
인슐린 분비 장애로 인한 hyperinsulinemia와 hypoglycemia를 감별합니다.
신장 기능 저하 시 proteinuria와 hematuria를 모니터링합니다.
subsegmental PE는 anticoagulation 없이도 추적 가능합니다.
bradycardia는 vagal tone 증가나 β-blocker 과다에 의해 발생합니다.
Dermatology에서 psoriasis와 atopic dermatitis 감별 중요.
monoclonal antibody로 trastuzumab, rituximab이 대표적입니다.
GLP-1 agonist와 SGLT2 inhibitor 병용 시 체중감소 효과.
ACE inhibitor는 ARB와 함께 renoprotective effect가 있습니다.
DPP4 inhibitor는 저혈당 위험이 낮습니다.
TIA 환자에서 MRI, MRA 시행 후 antiplatelet therapy 고려.
macrophage activation syndrome에서 ferritin 수치가 급격히 상승.
pseudomembranous colitis는 C. difficile에 의해 발생합니다.
neoplasm의 staging을 위해 PET-CT와 biopsy가 필요합니다.
hemimegalencephaly는 드문 뇌발달 이상입니다.
"""

def run_test(text: str):
    normalized = unicodedata.normalize("NFC", text)

    print("=" * 65)
    print("  WHISPER INJECTION 키워드 추출 테스트")
    print("=" * 65)

    # ── 1. PREFIX 패턴 ────────────────────────────────────────────
    prefix_matches = list({m.group() for m in MEDICAL_PREFIX_RE.finditer(normalized)})
    print(f"\n[1] PREFIX 패턴 ({len(prefix_matches)}개):")
    print(f"    regex: {MEDICAL_PREFIX_RE.pattern[:60]}...")
    for w in sorted(prefix_matches):
        print(f"    + {w}")

    # ── 문제 탐지: prefix 내부에서 짧은 단어 누락 확인 ──────────────
    # IGNORECASE인데 [a-z]{3,} → 대문자 단어도 잡히는지
    test_words_prefix = ["Hypertension", "HYPERTENSION", "anti", "antimicrobial",
                         "preload", "postop", "dehydration", "pancreatitis"]
    print(f"\n    [PREFIX 개별 매치 확인]")
    for w in test_words_prefix:
        m = MEDICAL_PREFIX_RE.fullmatch(w)
        print(f"    fullmatch('{w}'): {'O' if m else 'X'}")

    # ── 2. SUFFIX 패턴 ────────────────────────────────────────────
    suffix_matches = list({m.group() for m in MEDICAL_SUFFIX_RE.finditer(normalized)})
    print(f"\n[2] SUFFIX 패턴 ({len(suffix_matches)}개):")
    print(f"    regex: {MEDICAL_SUFFIX_RE.pattern[:60]}...")
    for w in sorted(suffix_matches):
        print(f"    + {w}")

    # ── 문제 탐지: 순수 suffix 자체가 매치되는지 확인 ────────────────
    test_words_suffix = ["itis", "logy", "oma", "hepatitis", "lymphoma",
                         "neurology", "cardiomegaly", "ectomy", "appendectomy"]
    print(f"\n    [SUFFIX 개별 매치 확인]")
    for w in test_words_suffix:
        m = MEDICAL_SUFFIX_RE.fullmatch(w)
        print(f"    fullmatch('{w}'): {'O' if m else 'X'}")

    # ── 핵심 문제: suffix 단독 단어 매치 여부 ─────────────────────
    print(f"\n    [!! SUFFIX 단독 텍스트 내 매치 확인 !!]")
    solo_suffix_text = "itis logy oma osis pathy 환자에게 ectomy 수행"
    solo_hits = [m.group() for m in MEDICAL_SUFFIX_RE.finditer(solo_suffix_text)]
    print(f"    입력: '{solo_suffix_text}'")
    print(f"    매치: {solo_hits}  ← 단독 suffix 단어가 잡히면 오탐!")

    # ── 3. ABBREV 패턴 ────────────────────────────────────────────
    abbrev_matches = list({m.group() for m in ABBREV_RE.finditer(normalized)})
    print(f"\n[3] ABBREV 패턴 ({len(abbrev_matches)}개):")
    print(f"    regex: {ABBREV_RE.pattern}")
    print(f"    ※ re.IGNORECASE 플래그 없음 → 소문자 섞인 약어 누락 가능")
    for w in sorted(abbrev_matches):
        print(f"    + {w}")

    # ── 문제 탐지: 소문자 섞인 약어, 1글자 ──────────────────────────
    test_abbrevs = ["ECG", "MI", "AF", "DVT", "TIA", "CT", "PCR",
                    "HbA1c", "INR", "eGFR", "pH"]  # 소문자 섞임
    print(f"\n    [ABBREV 개별 매치 확인]")
    for w in test_abbrevs:
        m = ABBREV_RE.fullmatch(w)
        print(f"    fullmatch('{w}'): {'O' if m else 'X'}")

    # ── 5. 최종 top 40 (extract_medical_keywords 시뮬레이션, 영문 only) ──
    seen: set[str] = set()
    priority_terms: list[str] = []

    def _add(term: str):
        key = term.upper()
        if key not in seen:
            seen.add(key)
            priority_terms.append(term)

    for m in MEDICAL_PREFIX_RE.finditer(normalized):
        _add(m.group())
    for m in MEDICAL_SUFFIX_RE.finditer(normalized):
        _add(m.group())
    for m in ABBREV_RE.finditer(normalized):
        _add(m.group())

    top40 = priority_terms[:40]
    print(f"\n[5] 최종 Whisper Injection 프롬프트 (top 40 / 실제 {len(priority_terms)}개):")
    print(f"    {', '.join(top40)}")

    # ── 6. 잠재적 문제 요약 ────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  잠재적 문제 요약")
    print("=" * 65)

    issues = []

    # 단독 suffix 오탐
    if solo_hits:
        issues.append(f"SUFFIX 오탐: '{solo_hits}' — [a-z]{{3,}} 조건이 단독 suffix 허용")

    # ABBREV가 소문자 혼용 약어를 못 잡음
    missed_abbrevs = [w for w in ["HbA1c", "eGFR", "pH"] if not ABBREV_RE.fullmatch(w)]
    if missed_abbrevs:
        issues.append(f"ABBREV 누락: {missed_abbrevs} — 소문자 시작 약어 미지원")

    # ABBREV가 단일 대문자 단어를 잡을 가능성
    single_upper = [m.group() for m in ABBREV_RE.finditer("I A B The OK it")]
    if single_upper:
        issues.append(f"ABBREV 과탐: {single_upper} — 2자 이상이지만 일반 단어 포함 가능")

    # prefix 패턴이 prefix 자체만인 단어 매치 여부
    short_prefix_hits = [m.group() for m in MEDICAL_PREFIX_RE.finditer("pre post sub de neo pan")]
    if short_prefix_hits:
        issues.append(f"PREFIX 과탐: {short_prefix_hits} — [a-z]{{3,}} 미달이므로 정상 차단됨 (OK)")

    if issues:
        for i, iss in enumerate(issues, 1):
            print(f"  [{i}] {iss}")
    else:
        print("  특이 문제 없음")

    print()


if __name__ == "__main__":
    run_test(SAMPLE_TEXT)

    # 실제 족보.txt가 있으면 추가 테스트
    import os
    if os.path.exists("족보.txt"):
        print("\n\n[실제 족보.txt 테스트]")
        with open("족보.txt", "r", encoding="utf-8") as f:
            jokbo = f.read()
        print(f"족보 텍스트 크기: {len(jokbo)}자\n")
        run_test(jokbo)
    elif os.path.exists("족보.pdf"):
        print("\n\n[족보.pdf 감지 → PyMuPDF 필요, 직접 main.py로 테스트하세요]")
