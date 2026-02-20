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

# --- System prompt ---
SML_SYSTEM_PROMPT = (
    "You are an AI assistant that uses Semantic Markup Language (SML) to ground "
    "your reasoning. When provided with SML context, you MUST:\n"
    "1. First write a <thinking> block that references specific SML anchor tokens\n"
    "2. Then write a <response> block with your answer grounded in the SML facts\n"
    "NEVER skip the <thinking> block. Always reference SML entities by their anchor tokens."
)

# --- Teacher prompt template ---
TEACHER_PROMPT_TEMPLATE = (
    "You are a neurosymbolic reasoner. I will give you a User Prompt and a Semantic "
    "Fact Sheet (SML block). You must respond in EXACTLY this format:\n\n"
    "<thinking>\n"
    "SML entities identified: [list the entities from the SML block with their anchor tokens]\n"
    "SML relations: [list the relations and what they mean]\n"
    "Reasoning: [explain your reasoning, explicitly referencing the SML data]\n"
    "</thinking>\n"
    "<response>\n"
    "[Your answer to the user's question, grounded in the SML facts]\n"
    "</response>\n\n"
    "CRITICAL RULES:\n"
    "- Your response MUST be grounded in the SML context provided\n"
    "- You MUST reference specific SML anchor tokens in your thinking\n"
    "- If the SML says something that contradicts common knowledge, FOLLOW THE SML\n"
    "- The <thinking> block must be at least 2-3 sentences\n"
    "- Never skip the <thinking> block\n"
    "- If a relation uses NOT_ prefix (e.g., NOT_CapableOf), it means the entity CANNOT do that action\n\n"
    "User Prompt: {prompt}\n\n"
    "SML Context:\n{sml_block}\n\n"
    "Now write your <thinking> and <response> blocks:"
)

# --- ConceptNet config ---
CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
CONCEPTNET_MIN_WEIGHT = 1.0
