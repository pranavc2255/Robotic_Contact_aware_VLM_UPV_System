import json
from pathlib import Path
from typing import Any

from PIL import Image


def extract_json(text: str) -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    json_object_text = _extract_first_json_object(cleaned)
    if json_object_text is None:
        raise ValueError("No JSON object found in model response.")
    return json.loads(json_object_text)


def _extract_first_json_object(text: str) -> str | None:
    start_idx = text.find("{")
    if start_idx < 0:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return None


class QwenRuntime:
    def __init__(self, model_dir: str, load_4bit: bool = False, mock: bool = False) -> None:
        self.model_dir = model_dir
        self.load_4bit = bool(load_4bit)
        self.mock = bool(mock)
        self.model = None
        self.processor = None
        self.torch_module = None
        self.ready = False

    def load(self) -> None:
        if self.mock:
            self.ready = True
            return

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1536 * 28 * 28,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.load_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_dir, **model_kwargs)
        self.torch_module = torch
        self.ready = True

    def metadata(self) -> dict:
        return {
            "ready": self.ready,
            "mock": self.mock,
            "model_dir": self.model_dir,
            "load_4bit": self.load_4bit,
        }

    def infer(
        self,
        image_path: str | Path,
        prompt_text: str,
        max_new_tokens: int = 700,
        temperature: float = 0.0,
    ) -> str:
        if not self.ready:
            raise RuntimeError("QwenRuntime is not ready. Call load() first.")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        if self.mock:
            return self._mock_response(image_path=image_path, prompt_text=prompt_text)

        if self.model is None or self.processor is None or self.torch_module is None:
            raise RuntimeError("Real Qwen runtime was not loaded correctly.")

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generation_kwargs["temperature"] = float(temperature)

        with self.torch_module.inference_mode():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated = output[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _mock_response(self, image_path: Path, prompt_text: str) -> str:
        if "top_blocked" in prompt_text:
            result = {
                "anchors": [
                    {
                        "id": "A1",
                        "top_blocked": 0,
                        "bottom_blocked": 0,
                        "has_mortar_or_attached_material": 0,
                        "has_nail_or_foreign_object": 0,
                        "has_broken_or_jagged_edge": 0,
                        "overall_score": 86,
                        "reason": f"mock clean contact for {image_path.name}",
                    },
                    {
                        "id": "A2",
                        "top_blocked": 1,
                        "bottom_blocked": 0,
                        "has_mortar_or_attached_material": 1,
                        "has_nail_or_foreign_object": 0,
                        "has_broken_or_jagged_edge": 0,
                        "overall_score": 45,
                        "reason": "mock attached material blocks one side",
                    },
                ],
                "ranking_best_to_worst": ["A1", "A2"],
                "best_anchor": "A1",
                "best_anchor_reason": "mock mode selects A1 deterministically",
            }
            return json.dumps(result)

        prompt_marker = "contact_flags" if "planar_contact_blocked" in prompt_text else "baseline"
        result = {
            "anchors": [
                {
                    "id": "A1",
                    "top_score": 88,
                    "bottom_score": 84,
                    "overall_score": 86,
                    "issues": ["mock response"],
                    "reason": f"deterministic mock response for {image_path.name} using {prompt_marker}",
                },
                {
                    "id": "A2",
                    "top_score": 72,
                    "bottom_score": 70,
                    "overall_score": 71,
                    "issues": ["mock lower score"],
                    "reason": "deterministic mock comparison",
                },
            ],
            "ranking_best_to_worst": ["A1", "A2"],
            "best_anchor": "A1",
            "best_anchor_reason": "mock mode selects A1 deterministically",
        }
        return json.dumps(result)
