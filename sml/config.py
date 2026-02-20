import os
from pathlib import Path

# Load .env file from project root if it exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# --- Path constants ---
DATA_DIR = PROJECT_ROOT / "data"
BIBLE_DB_PATH = DATA_DIR / "sml_bible.db"
TRAINING_DATA_PATH = DATA_DIR / "training_data.jsonl"
MODEL_OUTPUT_DIR = DATA_DIR / "model_output"

# --- SML Schema constants ---
EDA_WIDTH = 8  # Entity Descriptor Array width
RA_WIDTH = 6   # Relation Array width

EDA_SLOTS = [
    "domain",
    "category",
    "subcategory",
    "specificity",
    "identity",
    "modifier_1",
    "modifier_2",
    "confidence",
]

RA_SLOTS = [
    "rel_type",
    "subject_ref",
    "object_ref",
    "weight",
    "temporal",
    "negation",
]

# --- Domain mapping ---
DOMAINS = {
    1: "physical",
    2: "abstract",
    3: "digital",
    4: "event",
    5: "fiction",
}

# --- ConceptNet relation type mapping (IDs 1-34) ---
RELATION_TYPES = {
    1: "IsA",
    2: "PartOf",
    3: "HasA",
    4: "HasProperty",
    5: "CapableOf",
    6: "AtLocation",
    7: "Causes",
    8: "HasPrerequisite",
    9: "HasFirstSubevent",
    10: "HasLastSubevent",
    11: "MotivatedByGoal",
    12: "UsedFor",
    13: "CreatedBy",
    14: "DefinedAs",
    15: "SymbolOf",
    16: "MadeOf",
    17: "ReceivesAction",
    18: "Desires",
    19: "CausesDesire",
    20: "HasContext",
    21: "SimilarTo",
    22: "Antonym",
    23: "DerivedFrom",
    24: "RelatedTo",
    25: "FormOf",
    26: "EtymologicallyRelatedTo",
    27: "Synonym",
    28: "MannerOf",
    29: "LocatedNear",
    30: "HasContext",
    31: "dbpedia/genre",
    32: "dbpedia/occupation",
    33: "dbpedia/language",
    34: "dbpedia/capital",
}

RELATION_TYPES_INV = {v: k for k, v in RELATION_TYPES.items()}

# --- Training hyperparameters ---
DEFAULT_TRAINING_ARGS = {
    "model_name": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "max_seq_length": 4096,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
}

# --- Groq config ---
GROQ_CONFIG = {
    "model": os.environ.get("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"),
    "max_tokens": 4096,
    "temperature": 0.7,
}

# --- System prompt ---
SML_SYSTEM_PROMPT = (
    "You are an AI assistant that uses Semantic Markup Language (SML) to ground "
    "your reasoning. When provided with SML context, use it to inform your thinking "
    "and ensure your response is accurate and well-grounded."
)

# --- Teacher prompt template ---
TEACHER_PROMPT_TEMPLATE = (
    "You are a neurosymbolic reasoner. I will give you a User Prompt and a Semantic "
    "Fact Sheet (SML block). You must write a response that uses a <thinking> block "
    "to reference the SML facts before answering in a <response> block. Ensure your "
    "reasoning is grounded solely in the SML provided.\n\n"
    "User Prompt: {prompt}\n\n"
    "SML Context:\n{sml_block}\n\n"
    "Now write your <thinking> and <response> blocks:"
)

# --- ConceptNet config ---
CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
CONCEPTNET_MIN_WEIGHT = 1.0
