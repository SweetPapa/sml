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
    "lora_r": 128,
    "lora_alpha": 256,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1.5e-4,
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

# --- Groq parallelization config (loaded from .env with Developer tier defaults) ---
GROQ_PARALLEL = {
    "max_concurrent": int(os.environ.get("GROQ_MAX_CONCURRENT", "15")),
    "rpm_target": int(os.environ.get("GROQ_RPM_TARGET", "100")),
    "tpm_budget": int(os.environ.get("GROQ_TPM_BUDGET", "80000")),
    "max_retries": int(os.environ.get("GROQ_MAX_RETRIES", "5")),
    "initial_backoff_s": 1.0,
}

# --- System prompt ---
SML_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to a structured knowledge base "
    "encoded in SML (Semantic Markup Language). When you receive a question, you "
    "will also receive an SML context block containing relevant entities and "
    "relationships.\n\n"
    "Your process:\n"
    "1. Write a <thinking> block where you interpret the SML data and reason "
    "through the answer. Reference entity anchors and relationships here to "
    "show your work.\n"
    "2. Write a <response> block with a clear, natural-language answer. This is "
    "what the user sees — write conversationally as a knowledgeable assistant. "
    "NEVER mention SML, entity IDs, anchor tokens, confidence scores, or "
    "relation type names in your response.\n\n"
    "The <thinking> block is your internal scratchpad. The <response> block is "
    "your public answer."
)

# --- Teacher prompt template ---
TEACHER_PROMPT_TEMPLATE = (
    "You are generating training data for a neurosymbolic AI. Given a user "
    "question and an SML fact sheet, produce a response in two parts.\n\n"
    "<thinking>\n"
    "Analyze the SML entities and relations. What do the anchors and "
    "relationships tell you? Reason through the answer step by step, "
    "referencing specific anchors (e.g., dog_2451, bark_15662) and relation "
    "types (e.g., CapableOf, AtLocation). If the SML data is thin, combine "
    "it with commonsense reasoning.\n"
    "</thinking>\n"
    "<response>\n"
    "Answer the user's question in plain, natural language. Be conversational "
    "and helpful — like a knowledgeable friend. NEVER mention SML, entity IDs "
    "(like dog_2451), anchor tokens, confidence scores, relation names (like "
    "IsA or CapableOf), or any technical markup in this section. The user has "
    "no idea SML exists.\n"
    "</response>\n\n"
    "RULES:\n"
    "- The <thinking> block MUST reference specific SML anchor tokens and "
    "explain your reasoning\n"
    "- The <response> block MUST be pure natural language — no SML jargon "
    "whatsoever\n"
    "- If the SML doesn't fully answer the question, use the SML data you DO "
    "have plus commonsense knowledge\n"
    "- Never say \"the SML context does not contain...\" — always give your "
    "best answer\n"
    "- Vary your thinking style — don't always start the same way\n"
    "- If a relation uses NOT_ prefix (e.g., NOT_CapableOf), it means the "
    "entity CANNOT do that action\n\n"
    "User Question: {prompt}\n\n"
    "SML Context:\n{sml_block}"
)

# --- ConceptNet config ---
CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
CONCEPTNET_MIN_WEIGHT = 1.0
