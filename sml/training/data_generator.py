"""Training data generator — inverted pipeline using SML Encoder + Groq."""
import json
import time
from pathlib import Path
from typing import Optional

from sml.config import (
    GROQ_CONFIG,
    SML_SYSTEM_PROMPT,
    TEACHER_PROMPT_TEMPLATE,
    TRAINING_DATA_PATH,
)


# Micro-PoC prompt set — hand-crafted prompts for the ~50 micro concepts
MICRO_PROMPTS = [
    "What color is the sun?",
    "Where do dogs like to go?",
    "Can birds fly?",
    "What do cats do when they are happy?",
    "Where do fish live?",
    "What is a chair used for?",
    "Do dogs bark?",
    "What color are apples?",
    "Where do children go during the day?",
    "Is the sun hot or cold?",
    "What do dogs like to do?",
    "Can fish swim?",
    "Where might you find a table?",
    "What is a book used for?",
    "Are cats big or small?",
    "What color is the sky?",
    "Do birds sit on trees?",
    "What is bread made of?",
    "Can dogs run fast?",
    "Where do cats live?",
    "What do children like to do?",
    "Is water hot or cold?",
    "The dog sat on the mat. Where is the dog?",
    "What animals can swim?",
    "What color is grass?",
    "The child is at school. What is the child doing?",
    "What is love?",
    "Can a person feel fear?",
    "What animals bark?",
    "The big brown dog ran in the park. Describe the scene.",
    "Is a bird a type of animal?",
    "What do you need water for?",
    "The cat sat on the chair. Where is the cat?",
    "What is milk?",
    "Can a car fly?",
    "What color is snow?",
    "Do fish live in water?",
    "What is the opposite of hot?",
    "The red ball is in the park. What color is the ball?",
    "Is a dog fast or slow?",
    "Where might you find a book?",
    "What does the sun look like?",
    "The small black cat is sleeping. What is the cat doing?",
    "What are houses used for?",
    "Can cats purr?",
    "What is knowledge?",
    "The bird flew over the house. Can birds fly?",
    "Is an apple a fruit?",
    "What color can a dog be?",
    "What do you feel when something scary happens?",
]


def generate_training_data(
    bible_path: str,
    groq_api_key: str,
    output_path: Optional[str] = None,
    prompts: Optional[list[str]] = None,
    num_examples: int = 200,
    spacy_model: str = "en_core_web_sm",
) -> str:
    """Generate training data using the inverted pipeline.

    1. Take prompts
    2. Run each through SML Encoder -> deterministic <sml> block
    3. Send prompt + SML block to Groq -> get <thinking> + <response>
    4. Assemble full training tuple in ChatML format

    Returns path to the output JSONL file.
    """
    from groq import Groq
    from tqdm import tqdm

    from sml.encoder.encoder import SMLEncoder

    output_path = output_path or str(TRAINING_DATA_PATH)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Use micro prompts or provided prompts
    if prompts is None:
        prompts = MICRO_PROMPTS

    # Limit to num_examples (cycle prompts if needed)
    if len(prompts) < num_examples:
        full_prompts = []
        while len(full_prompts) < num_examples:
            full_prompts.extend(prompts)
        prompts = full_prompts[:num_examples]
    else:
        prompts = prompts[:num_examples]

    # Initialize encoder and Groq client
    encoder = SMLEncoder(bible_path, spacy_model=spacy_model)
    client = Groq(api_key=groq_api_key)

    generated = 0
    failed = 0

    with open(output_path, "w") as f:
        for prompt in tqdm(prompts, desc="Generating training data"):
            try:
                # Step 1: Encode the prompt into SML
                sml_block = encoder.encode(prompt)

                # Step 2: Send to Groq for thinking + response
                teacher_prompt = TEACHER_PROMPT_TEMPLATE.format(
                    prompt=prompt, sml_block=sml_block
                )

                completion = client.chat.completions.create(
                    model=GROQ_CONFIG["model"],
                    messages=[
                        {"role": "system", "content": "You are a neurosymbolic AI reasoning assistant. You must ground all your reasoning in the provided SML (Semantic Markup Language) context."},
                        {"role": "user", "content": teacher_prompt},
                    ],
                    max_tokens=GROQ_CONFIG["max_tokens"],
                    temperature=GROQ_CONFIG["temperature"],
                )

                teacher_response = completion.choices[0].message.content
                if not teacher_response:
                    failed += 1
                    continue

                # Step 3: Ensure response has proper tags
                thinking, response = _parse_teacher_response(teacher_response)
                if not thinking or not response:
                    failed += 1
                    continue

                # Step 4: Assemble training tuple in ChatML format
                assistant_content = f"{sml_block}\n<thinking>\n{thinking}\n</thinking>\n<response>\n{response}\n</response>"

                training_example = {
                    "messages": [
                        {"role": "system", "content": SML_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }

                f.write(json.dumps(training_example) + "\n")
                generated += 1

            except Exception as e:
                print(f"\nError on prompt '{prompt[:50]}...': {e}")
                failed += 1

            # Rate limiting — Groq free tier is ~30 req/min
            time.sleep(2.0)

    encoder.close()
    print(f"\nGeneration complete: {generated} examples, {failed} failures")
    print(f"Output: {output_path}")
    return output_path


def _parse_teacher_response(response: str) -> tuple[str, str]:
    """Extract thinking and response content from the teacher model output."""
    thinking = ""
    answer = ""

    # Try to extract <thinking> block
    if "<thinking>" in response and "</thinking>" in response:
        start = response.index("<thinking>") + len("<thinking>")
        end = response.index("</thinking>")
        thinking = response[start:end].strip()

    # Try to extract <response> block
    if "<response>" in response and "</response>" in response:
        start = response.index("<response>") + len("<response>")
        end = response.index("</response>")
        answer = response[start:end].strip()

    # Fallback: if no tags, treat the whole thing as the response
    if not thinking and not answer:
        # Split at a reasonable point
        lines = response.strip().split("\n")
        if len(lines) > 2:
            thinking = "\n".join(lines[:len(lines)//2])
            answer = "\n".join(lines[len(lines)//2:])
        else:
            return "", ""

    return thinking, answer
