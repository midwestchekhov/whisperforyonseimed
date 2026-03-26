# medical_dicts.py
# 다층 의학용어 딕셔너리 - Whisper postprocessing Layer a/b에서 사용

# ── Layer a/b: 포네틱 alias (Whisper 오인식 한글 → 의학용어 후보) ─────────────
# 키: Whisper가 잘못 전사할 수 있는 한글 표현
# 값: 교정 후보 리스트 (첫 번째가 최우선 교정값)
PHONETIC_ALIASES: dict[str, list[str]] = {

    # ── Dermatology ──────────────────────────────────────────────────────────
    "초코":         ["수포"],           # vesicle 오인식
    "수표":         ["수포"],
    "초코전":       ["수포진"],
    "포진":         ["수포진"],
    "반진":         ["발진"],           # rash
    "가리움":       ["가려움"],         # pruritus
    "홍바":         ["홍반"],           # erythema
    "팬피거스":     ["천포창"],         # pemphigus
    "팸피거스":     ["천포창"],
    "팸피고이드":   ["수포성 유사천포창"],  # pemphigoid
    "팬피고이드":   ["수포성 유사천포창"],
    "소리아시스":   ["건선"],           # psoriasis
    "건선증":       ["건선"],
    "아토피":       ["아토피피부염"],
    "습진":         ["습진"],
    "두드리기":     ["두드러기"],       # urticaria
    "두들기기":     ["두드러기"],
    "멜라노마":     ["흑색종"],         # melanoma
    "기저세포":     ["기저세포암"],     # BCC
    "편평세포":     ["편평세포암"],     # SCC
    "색소":         ["색소침착"],

    # ── Pharmacology ─────────────────────────────────────────────────────────
    "지피":         ["DPP"],            # DPP4 inhibitor
    "지피피":       ["DPP4"],
    "지피사":       ["DPP4"],
    "지엘피":       ["GLP"],            # GLP-1
    "지엘피원":     ["GLP-1"],
    "에스지엘티":   ["SGLT2"],
    "에스지엘티투": ["SGLT2"],
    "에이스":       ["ACE"],            # ACE inhibitor
    "에이씨이":     ["ACE"],
    "에이알비":     ["ARB"],            # ARB
    "엔세이드":     ["NSAID"],
    "엔사이드":     ["NSAID"],
    "베타차단":     ["β-blocker"],
    "베타블로커":   ["β-blocker"],
    "칼슘차단":     ["CCB"],            # calcium channel blocker
    "칼시움차단":   ["CCB"],
    "스타틴":       ["statin"],
    "메트포민":     ["metformin"],
    "인슐린":       ["insulin"],
    "헤파린":       ["heparin"],
    "와파린":       ["warfarin"],
    "아스피린":     ["aspirin"],
    "클로피":       ["clopidogrel"],
    "독소루":       ["doxorubicin"],
    "탁솔":         ["paclitaxel"],
    "빈크리스":     ["vincristine"],
    "시스플라틴":   ["cisplatin"],
    "메토트렉":     ["methotrexate"],
    "싸이클로":     ["cyclophosphamide"],
    "이마티닙":     ["imatinib"],
    "리툭시맙":     ["rituximab"],
    "트라스투주":   ["trastuzumab"],

    # ── Cardiology / Pulmonology ──────────────────────────────────────────────
    "심방세동":     ["심방세동"],       # AF - already correct but alias for variants
    "심방잔떨림":   ["심방세동"],
    "심실세동":     ["심실세동"],       # VF
    "심근경색":     ["심근경색"],       # MI
    "부정맥":       ["부정맥"],         # arrhythmia
    "허혈":         ["허혈"],           # ischemia
    "폐색전":       ["폐색전증"],       # PE
    "폐혈전":       ["폐색전증"],
    "심부전":       ["심부전"],         # HF
    "고혈압":       ["고혈압"],         # HTN
    "저혈압":       ["저혈압"],
    "협심증":       ["협심증"],         # angina

    # ── Endocrinology ────────────────────────────────────────────────────────
    "당뇨":         ["당뇨병"],         # DM
    "갑상선항진":   ["갑상선기능항진증"],    # hyperthyroidism
    "갑상선저하":   ["갑상선기능저하증"],    # hypothyroidism
    "쿠싱":         ["쿠싱증후군"],     # Cushing's
    "에디슨":       ["에디슨병"],       # Addison's
    "고지혈":       ["고지혈증"],       # hyperlipidemia

    # ── Nephrology ──────────────────────────────────────────────────────────
    "신부전":       ["신부전"],
    "사구체":       ["사구체염"],       # GN
    "네프로틱":     ["신증후군"],       # nephrotic syndrome

    # ── Oncology ────────────────────────────────────────────────────────────
    "악성":         ["악성종양"],
    "양성":         ["양성종양"],
    "암성":         ["악성종양"],

    # ── Common STT artifacts ──────────────────────────────────────────────────
    "되서":         ["돼서"],           # grammar
    "않":           ["않"],
    "dp":           ["DPP4"],
    "gp":           ["GLP"],
}


# ── Layer a: 일반 의학 어휘 (알려진 용어 = 의심 대상 제외) ────────────────────
# 이 목록에 있으면 Layer a에서 suspicious로 분류하지 않음
GENERAL_MEDICAL_TERMS: list[str] = [
    # Prefix-based terms
    "hyperglycemia", "hypoglycemia", "hyperkalemia", "hypokalemia",
    "hypertension", "hypotension", "hyperthermia", "hypothermia",
    "hyperthyroidism", "hypothyroidism", "hyperplasia", "hypoplasia",
    "tachycardia", "bradycardia", "tachypnea", "bradypnea",
    "polyneuropathy", "mononeuropathy", "polyuria", "polydipsia",
    "microangiopathy", "macroangiopathy", "microcephaly", "macrocephaly",
    "neoplasm", "neoplasia", "neonatal", "neonatology",
    "pseudomembrane", "pseudocyst", "pseudogout",
    "pancreatitis", "pancytopenia", "panhypopituitarism",
    "hemiplasia", "hemiplegia", "hemiparesis",
    "antihypertensive", "anticoagulant", "antibiotic", "antifungal",
    "antiviral", "antiemetic", "antidepressant", "antipsychotic",
    "preoperative", "postoperative", "intraoperative", "perioperative",
    "subcutaneous", "intramuscular", "intravenous", "intradermal", "intrathecal",
    "interstitial", "intracellular", "intercellular", "extracellular",
    "dehydration", "decompensation", "demyelination", "decortication",
    "bradykinesia", "tachyphylaxis",

    # Suffix-based conditions
    "appendicitis", "gastritis", "hepatitis", "nephritis", "arthritis",
    "dermatitis", "colitis", "rhinitis", "sinusitis", "bronchitis",
    "pancreatitis", "endocarditis", "pericarditis", "myocarditis",
    "encephalitis", "meningitis", "osteomyelitis", "cellulitis",
    "carcinoma", "lymphoma", "melanoma", "sarcoma", "glioma", "adenoma",
    "fibroma", "lipoma", "neuroma", "hemangioma", "teratoma",
    "fibrosis", "cirrhosis", "stenosis", "thrombosis", "embolism",
    "sclerosis", "atherosclerosis", "arteriosclerosis",
    "neuropathy", "myopathy", "cardiomyopathy", "nephropathy", "retinopathy",
    "hepatopathy", "encephalopathy", "gastropathy",
    "pathology", "histology", "cytology", "pharmacology", "immunology",
    "cardiology", "neurology", "dermatology", "oncology", "hematology",
    "nephrology", "pulmonology", "endocrinology", "rheumatology",
    "appendectomy", "cholecystectomy", "nephrectomy", "mastectomy",
    "gastrectomy", "colectomy", "thyroidectomy", "splenectomy",
    "angioplasty", "rhinoplasty", "arthroplasty",
    "colonoscopy", "endoscopy", "bronchoscopy", "laparoscopy",
    "cystoscopy", "hysteroscopy", "arthroscopy",
    "laparotomy", "craniotomy", "thoracotomy",
    "colostomy", "tracheostomy", "ileostomy",
    "angiography", "mammography", "radiography", "sonography",
    "electrocardiogram", "echocardiogram", "electroencephalogram",
    "anemia", "leukemia", "uremia", "septicemia", "bacteremia",
    "hematuria", "proteinuria", "glycosuria", "pyuria",
    "neuralgia", "myalgia", "arthralgia", "cardialgia",
    "hypertrophy", "atrophy", "dystrophy",
    "pathogenesis", "carcinogenesis", "neurogenesis", "angiogenesis",
    "dysplasia", "metaplasia", "aplasia",
    "splenomegaly", "hepatomegaly", "cardiomegaly", "lymphadenopathy",

    # Common abbreviations
    "ACE", "ARB", "NSAID", "DPP4", "GLP1", "SGLT2",
    "CBC", "BMP", "CMP", "LFT", "RFT", "ABG", "LDH",
    "ECG", "EKG", "MRI", "CT", "PET", "CXR", "USS",
    "IV", "IM", "SC", "PO", "PRN", "QD", "BID", "TID", "QID",
    "ICU", "CCU", "ER", "OR", "OPD",
    "HTN", "DM", "CAD", "CHF", "COPD", "CKD", "AKI",
    "MI", "PE", "DVT", "CVA", "TIA", "SAH", "SDH",
    "HIV", "HBV", "HCV", "HPV", "CMV", "EBV", "HSV", "VZV",
    "TNF", "IL", "IFN", "TGF", "VEGF", "EGFR",
    "RBC", "WBC", "PLT", "Hgb", "Hct", "MCV", "MCH",
    "Na", "K", "Cl", "Ca", "Mg", "PO4", "HCO3",
    "BUN", "Cr", "eGFR", "ALT", "AST", "GGT", "ALP",
    "INR", "PTT", "PT", "aPTT", "TT",
    "TSH", "T3", "T4", "FT4", "LH", "FSH", "ACTH", "GH",
    "PSA", "CEA", "AFP", "CA125", "CA19-9",
    "PCR", "ELISA", "WES", "NGS",
    "BMI", "BP", "HR", "RR", "SpO2", "GCS",
    "COPD", "ARDS", "CHF", "ACS", "STEMI", "NSTEMI",
    "HbA1c", "FPG", "OGTT",

    # Korean common terms (자주 쓰이는 한국어 의학용어)
    "고혈압", "당뇨병", "심근경색", "심부전", "폐렴", "패혈증",
    "갑상선", "부신", "뇌졸중", "협심증", "관상동맥",
    "신부전", "간경변", "복막염", "충수염", "담낭염",
    "골절", "탈구", "염좌", "타박상",
    "수술", "마취", "절개", "봉합", "지혈",
    "항생제", "항바이러스제", "항진균제", "항암제",
    "면역억제제", "스테로이드", "코르티코스테로이드",
    "혈전", "색전", "허혈", "괴사", "섬유화", "경화",
    "악성종양", "양성종양", "전이", "침윤",
    "자가면역", "알레르기", "과민반응", "아나필락시스",
    "염증", "부종", "삼출", "농양", "육아종",
    "빈혈", "백혈병", "혈소판감소증", "혈우병",
    "두드러기", "발진", "홍반", "수포", "농포", "구진",
    "가려움", "소양증",
]


# ── Layer a: 과목별 특이 의학용어 ──────────────────────────────────────────────
SUBJECT_SPECIFIC_TERMS: dict[str, list[str]] = {
    "histology": [
        "epithelium", "simple cuboidal epithelium", "simple columnar epithelium",
        "stratified squamous epithelium", "germinal epithelium", "surface epithelium",
        "tunica albuginea", "cortex", "medulla", "dense irregular connective tissue",
        "mucosa", "muscularis", "serosa", "ciliated cell", "peg cell",
        "follicle", "primordial follicle", "primary follicle", "secondary follicle",
        "graafian follicle", "follicular stigma", "granulosa cell", "theca layer",
        "theca interna", "theca externa", "zona pellucida", "antrum",
        "primary oocyte", "secondary oocyte", "corpus luteum", "corpus albicans",
        "luteinization", "cumulus oophorus", "corona radiata", "ovary", "uterine tube",
        "histology", "histo", "gland", "stroma", "parenchyma",
        "상피", "단층입방상피", "단층원주상피", "중층편평상피", "표면상피", "배상피",
        "치밀불규칙결합조직", "피질", "수질", "점막", "근육층", "장막",
        "원시난포", "일차난포", "이차난포", "성숙난포", "과립막세포", "난포막",
        "난포막내층", "난포막외층", "투명대", "난포강", "일차난모세포",
        "이차난모세포", "황체", "백체", "황체화", "난소", "난관", "조직학",
    ],

    "embryology": [
        "fertilization", "capacitation", "acrosome", "acrosome reaction",
        "cortical reaction", "polyspermy", "blastocyst", "inner cell mass", "ICM",
        "trophoblast", "cytotrophoblast", "syncytiotrophoblast", "implantation",
        "decidua", "decidua basalis", "decidua capsularis", "decidua parietalis",
        "chorion", "chorionic villi", "lacuna", "intervillous space",
        "umbilical vesicle", "yolk sac", "amnion", "amniotic cavity",
        "extraembryonic mesoderm", "somatopleuric mesoderm", "splanchnopleuric mesoderm",
        "gastrulation", "primitive streak", "notochord", "epiblast", "hypoblast",
        "ectoderm", "mesoderm", "endoderm", "organogenesis", "placenta",
        "placental membrane", "connecting stalk", "umbilical cord",
        "fetal period", "previable fetus", "viable fetus",
        "수정", "수정란", "수정능획득", "첨체", "첨체반응", "피질반응", "다정자수정",
        "포배", "배반포", "내세포괴", "영양막", "세포영양막", "합포영양막", "착상",
        "탈락막", "바닥탈락막", "피막탈락막", "벽탈락막", "융모막", "융모", "강",
        "융모간강", "배꼽소포", "난황낭", "양막", "양막강", "배외중배엽", "체벽중배엽",
        "내장중배엽", "원선", "원시선", "척삭", "외배엽", "중배엽", "내배엽",
        "기관형성", "태반", "태반막", "연결줄기", "제대", "태아기", "생존불가능태아",
    ],

    "dermatology": [
        # Lesion morphology
        "papule", "macule", "patch", "plaque", "nodule", "tumor",
        "vesicle", "bulla", "pustule", "wheal", "crust", "scale",
        "erosion", "ulcer", "fissure", "atrophy", "scar", "keloid",
        "telangiectasia", "purpura", "petechiae", "ecchymosis",
        "erythema", "cyanosis", "jaundice", "hyperpigmentation", "hypopigmentation",
        # Conditions
        "urticaria", "angioedema", "pruritus", "xerosis",
        "alopecia", "hirsutism", "onychomycosis", "tinea",
        "psoriasis", "eczema", "atopic dermatitis", "contact dermatitis",
        "seborrheic dermatitis", "rosacea", "acne vulgaris",
        "pemphigus", "pemphigoid", "bullous pemphigoid",
        "epidermolysis bullosa", "dermatitis herpetiformis",
        "Steven-Johnson syndrome", "toxic epidermal necrolysis",
        "melanoma", "basal cell carcinoma", "squamous cell carcinoma",
        "Kaposi sarcoma", "mycosis fungoides",
        "vitiligo", "melasma", "nevus",
        "herpes zoster", "herpes simplex", "varicella",
        "impetigo", "cellulitis", "erysipelas", "folliculitis",
        # Korean
        "수포", "농포", "구진", "반점", "태선화", "가피", "인설",
        "두드러기", "가려움", "소양감", "발진", "홍반", "자반",
        "흑색종", "기저세포암", "편평세포암", "건선", "습진",
        "천포창", "유사천포창", "대상포진", "단순포진", "농가진",
    ],

    "pharmacology": [
        # PK/PD basics
        "agonist", "antagonist", "partial agonist", "inverse agonist",
        "receptor", "ligand", "efficacy", "potency", "affinity",
        "pharmacokinetics", "pharmacodynamics",
        "absorption", "distribution", "metabolism", "excretion",
        "bioavailability", "half-life", "clearance", "volume of distribution",
        "first-pass effect", "enterohepatic circulation",
        "cytochrome", "CYP3A4", "CYP2D6", "CYP1A2",
        "LD50", "ED50", "therapeutic index", "therapeutic window",
        "tolerance", "dependence", "withdrawal", "tachyphylaxis",
        # Drug classes
        "ACE inhibitor", "ARB", "beta-blocker", "calcium channel blocker",
        "diuretic", "thiazide", "furosemide", "spironolactone",
        "statin", "fibrate", "ezetimibe",
        "NSAID", "opioid", "acetaminophen", "salicylate",
        "benzodiazepine", "barbiturate", "GABA",
        "antipsychotic", "antidepressant", "anxiolytic", "SSRI", "SNRI",
        "DPP4 inhibitor", "GLP-1 agonist", "SGLT2 inhibitor",
        "metformin", "insulin", "thiazolidinedione", "sulfonylurea",
        "anticoagulant", "antiplatelet", "thrombolytic",
        "heparin", "warfarin", "DOAC", "rivaroxaban", "dabigatran",
        "aspirin", "clopidogrel", "ticagrelor",
        "tPA", "alteplase", "streptokinase",
        "antibiotic", "penicillin", "cephalosporin", "carbapenem",
        "fluoroquinolone", "macrolide", "aminoglycoside", "vancomycin",
        "antifungal", "amphotericin", "azole", "echinocandin",
        "antiviral", "acyclovir", "oseltamivir", "remdesivir",
        "immunosuppressant", "cyclosporine", "tacrolimus", "sirolimus",
        "corticosteroid", "prednisone", "dexamethasone",
        "chemotherapy", "doxorubicin", "paclitaxel", "cisplatin",
        "targeted therapy", "imatinib", "erlotinib", "trastuzumab",
        "immunotherapy", "checkpoint inhibitor", "PD-1", "PD-L1", "CTLA-4",
        # Korean
        "약동학", "약력학", "반감기", "생체이용률", "수용체", "리간드",
        "작용제", "길항제", "부분작용제", "역작용제",
        "치료역", "내성", "의존성", "금단", "부작용", "금기",
        "초회통과효과", "간장초회통과", "단백결합",
        "항생제", "항균제", "항바이러스제", "항진균제",
        "면역억제제", "스테로이드", "진통제",
    ],

    "anatomy": [
        # Directional terms
        "anterior", "posterior", "superior", "inferior", "medial", "lateral",
        "proximal", "distal", "superficial", "deep", "central", "peripheral",
        "ipsilateral", "contralateral", "bilateral",
        # Planes
        "coronal", "sagittal", "transverse", "axial", "oblique",
        # Movements
        "flexion", "extension", "abduction", "adduction", "rotation",
        "pronation", "supination", "inversion", "eversion",
        "protraction", "retraction", "elevation", "depression",
        # Heart layers
        "epicardium", "myocardium", "endocardium", "pericardium",
        # Other membranes
        "pleura", "peritoneum", "mesentery", "omentum", "fascia",
        # Organ parts
        "cortex", "medulla", "hilum", "capsule", "parenchyma", "stroma",
        # Vasculature
        "artery", "vein", "capillary", "lymphatic", "endothelium",
        "adventitia", "tunica media", "tunica intima",
        # Neuro
        "neuron", "axon", "dendrite", "synapse", "ganglion",
        "afferent", "efferent", "somatic", "autonomic",
        "sympathetic", "parasympathetic",
        # Korean
        "심장", "폐", "간", "신장", "비장", "췌장", "위", "소장", "대장",
        "동맥", "정맥", "모세혈관", "신경", "근육", "뼈", "연골",
        "뇌", "척수", "척추", "흉부", "복부", "골반",
        "복막", "흉막", "심낭", "장간막", "대망",
    ],

    "pathology": [
        # Cell death
        "necrosis", "apoptosis", "autophagy", "pyroptosis", "necroptosis",
        "coagulative necrosis", "liquefactive necrosis", "caseous necrosis",
        "fat necrosis", "gangrenous necrosis", "fibrinoid necrosis",
        # Inflammation
        "inflammation", "acute inflammation", "chronic inflammation",
        "granuloma", "abscess", "empyema", "fistula", "sinus",
        "exudate", "transudate", "edema", "effusion",
        "neutrophil", "macrophage", "lymphocyte", "plasma cell",
        "giant cell", "foreign body reaction",
        # Cellular adaptations
        "hyperplasia", "hypertrophy", "atrophy", "metaplasia", "dysplasia",
        "carcinoma in situ", "invasive carcinoma",
        # Tumor biology
        "benign", "malignant", "metastasis", "invasion", "angiogenesis",
        "lymphangiogenesis", "oncogene", "tumor suppressor",
        "TP53", "RB", "BRCA", "KRAS", "EGFR", "HER2",
        "epithelial-mesenchymal transition", "EMT",
        # Fibrosis / repair
        "fibrosis", "cirrhosis", "sclerosis", "scarring",
        "granulation tissue", "wound healing",
        # Hemodynamics
        "thrombosis", "embolism", "infarction", "ischemia", "reperfusion",
        "hyperemia", "congestion", "hemorrhage", "hemostasis",
        # Korean
        "괴사", "세포자멸사", "자가포식", "섬유화", "경화", "육아종",
        "악성", "양성", "전이", "침윤", "혈전", "색전", "경색",
        "부종", "삼출", "농양", "누공", "배농",
        "과형성", "비대", "위축", "화생", "이형성",
    ],

    "physiology": [
        # Homeostasis
        "homeostasis", "feedback", "negative feedback", "positive feedback",
        "set point", "steady state",
        # Electrophysiology
        "action potential", "resting potential", "membrane potential",
        "depolarization", "repolarization", "refractory period",
        "threshold", "all-or-none",
        # Cardiac physiology
        "cardiac output", "stroke volume", "heart rate", "ejection fraction",
        "preload", "afterload", "contractility", "compliance",
        "Frank-Starling", "Starling curve",
        "systole", "diastole", "systolic", "diastolic",
        "cardiac cycle", "Wiggers diagram",
        # Pulmonary
        "tidal volume", "vital capacity", "residual volume",
        "FEV1", "FVC", "FEV1/FVC", "TLC", "FRC",
        "ventilation", "perfusion", "V/Q ratio",
        "PaO2", "PaCO2", "SaO2", "SpO2",
        # Renal
        "glomerular filtration rate", "GFR", "renal plasma flow",
        "tubular reabsorption", "tubular secretion",
        "osmolarity", "osmolality", "tonicity", "oncotic pressure",
        # Acid-base
        "pH", "bicarbonate", "base excess", "anion gap",
        "metabolic acidosis", "metabolic alkalosis",
        "respiratory acidosis", "respiratory alkalosis",
        # Hormones
        "insulin", "glucagon", "cortisol", "aldosterone",
        "ADH", "ANP", "BNP", "renin", "angiotensin",
        "erythropoietin", "thrombopoietin", "EPO",
        "TSH", "T3", "T4", "PTH", "calcitonin",
        # Korean
        "심박출량", "일회박출량", "박출계수", "전부하", "후부하",
        "사구체여과율", "산염기균형", "삼투압", "종양압",
        "호르몬", "피드백", "항상성", "기저대사율",
    ],

    "immunology": [
        # Cells
        "lymphocyte", "T cell", "B cell", "NK cell", "dendritic cell",
        "macrophage", "neutrophil", "eosinophil", "basophil", "mast cell",
        "plasma cell", "memory cell", "regulatory T cell", "Treg",
        "CD4", "CD8", "CD20", "CD19", "CD3",
        # Molecules
        "antibody", "immunoglobulin", "IgG", "IgM", "IgA", "IgE", "IgD",
        "antigen", "epitope", "paratope", "hapten",
        "complement", "cytokine", "interleukin", "interferon",
        "TNF-alpha", "IL-1", "IL-6", "IL-17", "IL-23",
        "MHC", "HLA", "TCR", "BCR",
        # Mechanisms
        "innate immunity", "adaptive immunity", "humoral immunity",
        "cell-mediated immunity", "passive immunity", "active immunity",
        "opsonization", "phagocytosis", "ADCC",
        "hypersensitivity", "type I", "type II", "type III", "type IV",
        "autoimmunity", "tolerance", "anergy", "apoptosis",
        # Diseases
        "SLE", "rheumatoid arthritis", "Sjogren", "scleroderma",
        "anaphylaxis", "allergy", "asthma", "atopy",
        # Korean
        "면역", "항체", "항원", "보체", "사이토카인",
        "자가면역", "알레르기", "과민반응", "아나필락시스",
        "선천면역", "적응면역", "체액면역", "세포성면역",
    ],

    "obstetrics": [
        # Pregnancy / parity
        "nulligravida", "gravida", "nullipara", "primipara", "multipara",
        "perinatal period", "term neonate", "preterm labor", "postterm pregnancy",
        "low birthweight", "macrosomia",
        # Advanced obstetric anatomy / repair
        "obstetric anal sphincter injuries", "OASIS",
        "bulbospongiosus muscle", "superficial transverse perineal muscle",
        "external anal sphincter", "internal anal sphincter",
        "perineal membrane", "rectovaginal fascia", "ischial tuberosity",
        "midline episiotomy", "mediolateral episiotomy", "fourchette", "hymenal ring",
        "subcuticular stitch", "figure-of-eight stitch",
        "end-to-end repair", "overlapping repair",
        # Reproductive histology / ovary
        "ovary", "ovarian follicle", "primordial follicle", "primary oocyte",
        "secondary oocyte", "granulosa cell", "theca interna", "theca externa",
        "zona pellucida", "antrum", "graafian follicle", "follicular development",
        "ovulation", "follicular atresia", "corpus luteum", "corpus albicans",
        "luteinization", "cumulus oophorus", "corona radiata",
        "estrogen", "progesterone", "androstenedione", "hCG", "LH",
        "uterine tube", "uterine duct", "mucosal fold", "muscular layer", "serosa",
        # Fetal lie / presentation / position
        "fetal lie", "longitudinal lie", "transverse lie",
        "presentation", "cephalic presentation", "breech presentation",
        "face presentation", "brow presentation", "shoulder presentation",
        "position", "LOA", "ROP", "LSA", "RSP",
        "frank breech", "complete breech", "incomplete breech", "footling breech",
        # Mechanism of labor
        "engagement", "descent", "flexion", "internal rotation", "extension",
        "external rotation", "expulsion", "asynclitism",
        # Labor abnormalities
        "dystocia", "cephalopelvic disproportion", "CPD", "uterine rupture",
        "shoulder dystocia", "nuchal arm",
        # Procedures
        "amniotomy", "episiotomy", "breech extraction",
        "external cephalic version", "ECV",
        "internal podalic version",
        "operative vaginal delivery", "OVD",
        "forceps", "vacuum extraction", "VBAC", "TOLAC",
        "mauriceau maneuver", "modified prague maneuver", "pinard maneuver",
        "duhrssen incisions", "zavanelli maneuver", "piper forceps",
        "axillary traction", "posterior axilla sling traction",
        "mcroberts maneuver", "wood screw maneuver", "gaskin maneuver",
        "symphysiotomy", "cleidotomy",
        "bakri balloon", "b-lynch suture", "hypogastric artery ligation",
        "cesarean hysterectomy", "cervical ripening", "bishop score",
        "misoprostol", "oxytocin augmentation", "pudendal block",
        "total breech extraction", "partial breech extraction",
        "spontaneous breech delivery",
        # Complications
        "preeclampsia", "eclampsia", "HELLP syndrome",
        "placenta previa", "placental abruption", "PPROM",
        "oligohydramnios", "polyhydramnios", "chorioamnionitis",
        "tocolysis", "terbutaline", "ritodrine", "mirror syndrome",
        "uterine atony", "couvelaire uterus", "velamentous insertion", "vasa previa",
        "alloimmunization", "retained placenta", "uterine inversion",
        "placenta accreta", "placenta increta", "placenta percreta",
        "hydrops fetalis", "gestational diabetes mellitus", "GDM",
        "intrauterine growth restriction", "IUGR",
        "gestational trophoblastic disease", "hydatidiform mole", "choriocarcinoma",
        "monoamniotic twins", "twin-to-twin transfusion syndrome", "TTTS",
        "vanishing twin", "nonstress test", "NST", "contraction stress test", "CST",
        "biophysical profile", "BPP", "doppler velocimetry",
        # Korean
        "미임부", "임부", "미산부", "초산부", "경산부",
        "주산기", "만삭아", "조산", "과숙임신", "저체중출생아", "거대아",
        "산과적 항문괄약근 손상", "전정구해면체근", "구해면체근", "표층횡회음근",
        "외항문괄약근", "내항문괄약근", "회음막", "직장질근막", "좌골결절",
        "중앙 회음절개술", "중측방 회음절개술", "음순후교련", "처녀막환",
        "피하내 봉합", "8자형 봉합", "끝끝이 이음 수술", "중첩 수복술",
        "난소", "원시난포", "일차난모세포", "이차난모세포", "과립막세포",
        "난포막내층", "난포막외층", "투명대", "난포강", "성숙난포",
        "배란", "난포폐쇄", "황체", "백체", "황체화", "난관", "자궁관",
        "점막주름", "근육층", "장막", "에스트로겐", "프로게스테론",
        "태위", "태세", "향배", "두정위", "둔위", "진둔위", "완전둔위",
        "불완전둔위", "족위", "좌천골전위", "우천골후위",
        "진입", "하강", "굴곡", "내회전", "신전", "외회전", "만출", "부동고정",
        "난산", "아두골반불균형", "자궁파열", "견갑난산", "목뒤팔",
        "인공양막파열", "회음절개술", "둔위견인술", "외회전술",
        "수술적 질식분만", "겸자", "흡입만출술", "내태아회전술",
        "모리소 수기", "수정 프라그 수기", "피나르 수기", "뒤르센 절개",
        "자바넬리 수기", "파이퍼 집게", "겨드랑이 견인법", "후방 겨드랑이 슬링 견인법",
        "맥로버츠 수기", "우드 나사 수기", "가스킨 수기", "두덩결합절개술",
        "빗장뼈절개술", "바크리 풍선", "비린치 봉합술", "하복동맥 결찰술",
        "제왕절개 후 자궁절제술", "자궁경부 숙화", "비숍 점수", "미소프로스톨",
        "옥시토신 진통촉진", "음부신경 차단술", "전둔위 추출술", "부분둔위 추출술",
        "자연 둔위 분만",
        "자간전증", "자간증", "헬프 증후군", "전치태반", "태반조기박리",
        "만삭 전 조기양막파열", "양수과소증", "양수과다증", "융모양막염",
        "자궁수축억제법", "터부탈린", "리토드린", "거울 증후군", "자궁무력증",
        "쿠벨레르 자궁", "막부착탯줄", "전치혈관", "동종면역", "잔류 태반",
        "자궁뒤집힘", "유착태반", "침윤태반", "투과태반", "태아수종",
        "임신당뇨병", "자궁내 성장지연", "임신영양막질환", "포상기태", "융모암",
        "일양막 쌍태아", "쌍태아 수혈증후군", "소실 쌍둥이",
        "비수축검사", "수축자극검사", "태아생물학적계수", "도플러 혈류계측법",
    ],

    "gynecology": [
        # Menstrual abnormalities
        "amenorrhea", "dysmenorrhea", "abnormal uterine bleeding", "AUB",
        "menorrhagia", "metrorrhagia", "PMS", "PMDD",
        # Major conditions
        "endometriosis", "adenomyosis", "leiomyoma",
        "pelvic inflammatory disease", "PID",
        "polycystic ovary syndrome", "PCOS",
        "ectopic pregnancy", "endometrial hyperplasia", "ovarian torsion",
        "borderline ovarian tumor", "gonadal dysgenesis",
        "premature ovarian insufficiency", "POI",
        # Mullerian / structural anomalies
        "mullerian anomaly", "bicornuate uterus", "septate uterus",
        "didelphic uterus", "vaginal agenesis",
        "transverse vaginal septum", "imperforate hymen", "mullerian agenesis",
        # Pelvic organ prolapse
        "pelvic organ prolapse", "rectocele", "cystocele", "enterocele",
        "POP-Q system", "stress urinary incontinence", "SUI",
        "detrusor overactivity", "colpocleisis", "sacrocolpopexy", "pessary",
        # Examination / tests
        "pap test", "colposcopy", "hysteroscopy", "laparoscopy",
        "hysterosalpingography", "speculum", "bimanual examination",
        "sentinel lymph node biopsy", "sentinel lymph node mapping",
        "urodynamic study", "cystometry", "urethral pressure profile",
        "acetic acid test", "schiller test", "bethesda system", "HPV DNA testing",
        # Other anatomy / symptoms
        "dyspareunia", "vaginismus", "vulvodynia",
        "adnexa", "vulva", "clitoris", "parametrium", "rectovaginal septum",
        "paravesical space", "pararectal space", "bartholin gland cyst",
        "adnexal mass", "torsion of adnexa", "nabothian cyst",
        # Procedures / oncology / fertility
        "pelvic exenteration", "cytoreductive surgery", "radical hysterectomy",
        "lymphadenectomy", "omentectomy", "neoadjuvant chemotherapy", "HIPEC",
        "midurethral sling", "sacral neuromodulation", "assisted reproductive technology",
        "ART", "in vitro fertilization", "IVF", "intracytoplasmic sperm injection", "ICSI",
        "controlled ovarian hyperstimulation", "COH", "ovarian hyperstimulation syndrome", "OHSS",
        "myomectomy", "endometrial ablation", "salpingectomy", "salpingostomy",
        "methotrexate", "MTX", "culdocentesis", "LEEP", "conization",
        "total abdominal hysterectomy", "TAH", "vaginal hysterectomy", "VH",
        "laparoscopic supracervical hysterectomy",
        # Additional pathology
        "cervical intraepithelial neoplasia", "CIN", "BRCA mutation", "lynch syndrome",
        "kallmann syndrome", "asherman syndrome", "hyperprolactinemia", "hirsutism",
        "adenocarcinoma", "squamous cell carcinoma", "clear cell carcinoma",
        "germ cell tumor", "granulosa cell tumor", "teratoma", "vulvar vestibulitis",
        "condyloma acuminatum", "tubo-ovarian abscess", "TOA", "bacterial vaginosis",
        "trichomoniasis", "candidiasis", "sarcoma botryoides", "CA-125", "meigs syndrome",
        "dermoid cyst", "endometrioma", "vesicovaginal fistula", "ureterovaginal fistula",
        "genitourinary syndrome of menopause",
        # Korean
        "무월경", "월경통", "비정상자궁출혈", "월경과다", "부정출혈",
        "월경전증후군", "월경전불쾌장애",
        "자궁내막증", "자궁선근증", "자궁근종", "골반염",
        "다낭난소증후군", "자궁외 임신", "자궁내막증식증", "난소염전",
        "경계성 난소종양", "생식샘발생장애", "조기난소부전",
        "뮐러관 기형", "쌍각자궁", "중격자궁", "중복자궁",
        "질결손", "횡질중격", "처녀막막힘증", "뮐러관발생불전",
        "골반장기탈출증", "직장류", "방광류", "소장류", "복압요실금",
        "배뇨근과활동성", "질폐쇄술", "천골질고정술", "페사리",
        "자궁경부세포검사", "질확대경검사", "자궁경검사", "복강경검사",
        "자궁난관조영술", "질경", "양수골반진찰",
        "성교통", "질경련", "외음부통", "자궁부속기", "외음부", "음핵",
        "감시림프절생검", "감시림프절 매핑", "요역동학검사", "방광내압측정술",
        "요도압측정법", "초산검사", "쉴러검사", "베데스다 체계", "인유두종바이러스 검사",
        "골반내장전절제술", "종양감축술", "광범위 자궁절제술", "림프절절제술",
        "그물막절제술", "선행화학요법", "복강내 온열항암화학요법", "요도중간슬링",
        "천수신경조절술", "보조생식술", "체외수정", "세포질내 정자주입술",
        "과배란유도", "난소과자극증후군", "자궁근종절제술", "자궁내막소생술",
        "난관절제술", "난관개구술", "메토트렉세이트", "더글라스와천자",
        "루프전기절제술", "원추절제술", "복식 전 자궁절제술", "질식 자궁절제술",
        "복강경하 경상부 자궁절제술", "자궁경부상피내종양", "BRCA 유전자 변이",
        "린치 증후군", "칼만증후군", "아셔만증후군", "고프로락틴혈증", "다모증",
        "선암", "편평세포암", "투명세포암", "생식세포종양", "과립막세포종양",
        "바르톨린샘낭종", "뾰족콘딜로마", "난관난소농양", "세균질증", "트리코모나스증",
        "칸디다증", "포도모양육종", "메이그스 증후군", "유피낭종", "자궁내막종",
        "나보트낭종", "자궁옆조직", "직장질중격", "방광질누공", "요관질누공",
        "폐경관련 비뇨생식기증후군",
    ],

    "pediatrics": [
        # Neonatal neuro / respiratory
        "neonatal encephalopathy", "HIE", "hypoxic-ischemic encephalopathy",
        "respiratory distress syndrome", "RDS",
        "bronchopulmonary dysplasia", "BPD",
        "transient tachypnea of the newborn", "TTN",
        "meconium aspiration syndrome", "MAS",
        "apnea of prematurity",
        # Neonatal GI / CV / other
        "necrotizing enterocolitis", "NEC",
        "intraventricular hemorrhage", "IVH",
        "patent ductus arteriosus", "PDA",
        "persistent pulmonary hypertension of the newborn", "PPHN",
        "hyperbilirubinemia", "kernicterus",
        "retinopathy of prematurity", "ROP",
        "neonatal abstinence syndrome", "NAS",
        "germinal matrix hemorrhage", "ventriculitis",
        "periventricular leukomalacia", "PVL", "sepsis neonatorum",
        "surfactant replacement therapy", "phototherapy", "exchange transfusion",
        # Pediatric diseases
        "BRUE", "brief resolved unexplained event",
        "SIDS", "sudden infant death syndrome",
        "cerebral palsy", "failure to thrive", "FTT",
        "cystic fibrosis", "intussusception", "volvulus", "croup",
        "bronchiolitis", "asthma", "sepsis", "meningitis",
        "pyloric stenosis", "hirschsprung disease", "meckel diverticulum",
        "biliary atresia", "nephrotic syndrome", "glomerulonephritis",
        "hemolytic uremic syndrome", "HUS", "type 1 diabetes mellitus",
        "diabetic ketoacidosis", "DKA", "growth hormone deficiency",
        "hypothyroidism", "juvenile idiopathic arthritis", "dermatomyositis",
        "thalassemia", "sickle cell anemia", "aplastic anemia",
        "idiopathic thrombocytopenic purpura", "acute lymphoblastic leukemia",
        "hodgkin lymphoma", "burkitt lymphoma", "osteosarcoma",
        "epiglottitis", "status asthmaticus", "foreign body aspiration",
        "rickets", "scurvy", "marasmus", "kwashiorkor", "encephalitis",
        "febrile seizure", "epilepsy", "west syndrome", "lennox-gastaut syndrome",
        "vesicoureteral reflux", "enuresis", "erythema infectiosum",
        "roseola infantum", "varicella", "pertussis", "mumps", "rubella",
        # Cardiac / congenital
        "congenital heart disease", "tetralogy of Fallot",
        "VSD", "ASD", "hydrocephalus", "microcephaly",
        "hypospadias", "cryptorchidism",
        "coarctation of the aorta", "transposition of the great arteries",
        "hypoplastic left heart syndrome", "kawasaki disease", "rheumatic fever",
        "infective endocarditis",
        # Neuro / muscle
        "muscular dystrophy", "spinal muscular atrophy", "SMA",
        "acrocyanosis", "central cyanosis", "spina bifida", "meningomyelocele",
        "guillain-barre syndrome", "wilms tumor", "neuroblastoma",
        "pervasive developmental disorder", "autism spectrum disorder",
        "attention deficit hyperactivity disorder", "phenylketonuria", "PKU",
        "galactosemia", "maple syrup urine disease", "G6PD deficiency",
        "down syndrome", "edwards syndrome", "patau syndrome",
        "turner syndrome", "klinefelter syndrome", "congenital adrenal hyperplasia",
        "henoch-schonlein purpura", "HSP", "systemic lupus erythematosus",
        "SLE", "hemophilia", "disseminated intravascular coagulation", "DIC",
        # Korean
        "신생아뇌병증", "저산소성허혈성뇌병증", "호흡곤란증후군",
        "기관지폐이형성증", "신생아일과성빈호흡", "태변흡인증후군",
        "미숙아무호흡", "신생아괴사성장염", "뇌실내출혈",
        "동맥관개존증", "신생아지속성폐동맥고혈압", "고빌리루빈혈증",
        "핵황달", "미숙아망막병증", "신생아금단증후군",
        "배아기질출혈", "뇌실염", "뇌실주위백질연화증", "신생아 패혈증",
        "계면활성제 보충요법", "광선요법", "교환수혈",
        "짧은분해불명사건", "영아돌연사증후군", "뇌성마비", "성장장애",
        "낭성섬유증", "장중첩증", "장염전", "크룹", "세기관지염",
        "천식", "패혈증", "수막염", "선천성심질환", "활로사징",
        "유문협착증", "히르슈슈프룽병", "메켈게실", "담도폐쇄증", "신증후군",
        "사구체신염", "용혈요독증후군", "1형 당뇨병", "당뇨병케토산증",
        "성장호르몬결핍증", "갑상샘저하증", "소아기 특발성 관절염", "피부근염",
        "지중해빈혈", "낫세포빈혈", "재생불량빈혈", "특발 혈소판감소 자반증",
        "급성 림프모구 백혈병", "호지킨 림프종", "버킷 림프종", "골육종",
        "후두개염", "천식지속상태", "이물흡인", "구루병", "괴혈병",
        "마라스무스", "콰시오커", "뇌염", "열성경련", "뇌전증",
        "웨스트 증후군", "레녹스-가스토 증후군", "방광요관역류", "유뇨증",
        "전염성 홍반", "돌발진", "수두", "백일해", "볼거리", "풍진",
        "대동맥축착", "대혈관전위", "좌심형성부전증후군", "가와사키병", "류마티스열",
        "감염성 심내막염", "이분척추", "수막척수탈출증", "길랑-바레 증후군",
        "윌름스종양", "신경모세포종", "전반적 발달장애", "자폐스펙트럼장애",
        "주의력결핍 과잉행동장애", "페닐케톤뇨증", "갈락토오스혈증", "단풍당뇨증",
        "다운 증후군", "에드워드 증후군", "파타우 증후군", "터너 증후군",
        "클라인펠터 증후군", "선천부신과다형성", "헤노흐-쇤라인 자반증",
        "전신홍반루푸스", "혈우병", "범발성 혈관내 응고",
        "심실중격결손", "심방중격결손", "수두증", "소두증",
        "요도하열", "잠복고환", "근이영양증", "척수성근위축증",
        "말단청색증", "중심성청색증",
    ],
}
