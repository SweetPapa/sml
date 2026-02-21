"""SML Inference Pipeline — full encode → model → response pipeline."""
import re
from typing import Optional


class SMLPipeline:
    """Full SML inference pipeline: text → encoder → SML → model → response."""

    def __init__(
        self,
        model_path: str,
        bible_path: str,
        spacy_model: str = "en_core_web_sm",
        max_new_tokens: int = 1024,
    ):
        from sml.encoder.encoder import SMLEncoder
        from sml.config import SML_SYSTEM_PROMPT

        self.encoder = SMLEncoder(bible_path, spacy_model=spacy_model)
        self.system_prompt = SML_SYSTEM_PROMPT
        self.max_new_tokens = max_new_tokens
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the fine-tuned model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def run(self, user_input: str, custom_sml: Optional[str] = None) -> dict:
        """Run the full pipeline on user input.

        Args:
            user_input: The user's question or prompt.
            custom_sml: Optional custom SML block to inject (for Liar Ablation testing).
                If None, the encoder generates SML automatically.

        Returns:
            dict with keys: sml_block, thinking, response, raw_output
        """
        import torch

        # Step 1: Generate SML block
        if custom_sml:
            sml_block = custom_sml
        else:
            sml_block = self.encoder.encode(user_input)

        # Step 2: Compose messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Prepend SML block and force <thinking> start in the assistant's turn
        prompt_text += sml_block + "\n<thinking>\n"

        # Step 3: Tokenize and generate
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Step 4: Parse the output — re-add <thinking> tag that was in the prompt
        full_output = sml_block + "\n<thinking>\n" + raw_output
        result = self._parse_output(full_output)
        result["sml_block"] = sml_block
        result["raw_output"] = raw_output

        return result

    # Regex for corrupted response tags the model sometimes produces
    _RESPONSE_TAG_RE = re.compile(r"<s?p?o?n?s?e?response>|<sresponse>|<sponse>")
    _RESPONSE_CLOSE_RE = re.compile(r"</s?p?o?n?s?e?response>|</sresponse>|</sponse>")

    def _parse_output(self, text: str) -> dict:
        """Parse model output into thinking and response sections."""
        thinking = ""
        response = ""

        # Extract <thinking> block
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()

        # Extract <response> block (exact tags)
        response_match = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
        if response_match:
            response = response_match.group(1).strip()

        # Fallback: thinking found but no clean <response> tags — take text after </thinking>
        if thinking and not response and "</thinking>" in text:
            after = text[text.index("</thinking>") + len("</thinking>"):].strip()
            response = after

        # Fallback: no tags at all — strip SML and use raw text
        if not response:
            clean = re.sub(r"<sml>.*?</sml>", "", text, flags=re.DOTALL).strip()
            response = clean

        # Clean corrupted response tags from the model output
        response = self._RESPONSE_TAG_RE.sub("", response)
        response = self._RESPONSE_CLOSE_RE.sub("", response)
        response = re.sub(r"</?response>", "", response)
        response = response.strip()

        return {"thinking": thinking, "response": response}

    def close(self):
        self.encoder.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
