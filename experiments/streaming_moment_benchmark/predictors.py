"""Late-query predictors. Model loading is delayed so protocol tests stay light."""
from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from .frames import FrameObservation
from .metrics import score_answer
from .protocol import Question


@dataclasses.dataclass(frozen=True)
class Prediction:
    text: str
    parsed_answer: str
    score: float
    metric: str

    def to_dict(self) -> dict[str, str | float]:
        return dataclasses.asdict(self)


def _incorrect_answer(question: Question) -> str:
    if question.answer_type == "multiple_choice":
        return next(key for key in question.choices if key != question.answer)
    return "insufficient evidence"


class EvidenceOraclePredictor:
    """Protocol oracle: correct iff all annotated required evidence survived."""

    def predict(self, question: Question, frames: Sequence[FrameObservation], *, evidence_available: bool) -> Prediction:
        del frames
        text = question.answer if evidence_available else _incorrect_answer(question)
        scored = score_answer(question, text)
        return Prediction(text, str(scored["parsed_answer"]), float(scored["score"]), "evidence_availability_oracle")


class QwenMultiImagePredictor:
    PROMPT_VERSION = "retained-frames-mcq-v2"

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "auto",
        max_new_tokens: int = 4,
        min_pixels: int | None = None,
        max_pixels: int | None = 256 * 28 * 28,
        cpu_threads: int = 16,
        interop_threads: int = 1,
    ) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(cpu_threads)
            try:
                torch.set_num_interop_threads(interop_threads)
            except RuntimeError:
                # PyTorch allows this setting only before the first parallel op.
                pass
        self._torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, use_fast=False)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if min_pixels is not None:
                image_processor.min_pixels = min_pixels
            if max_pixels is not None:
                image_processor.max_pixels = max_pixels
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype="auto",
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(device)
        self.model.eval()

    @staticmethod
    def _prompt(question: Question) -> str:
        if question.choices:
            choices = "\n".join(f"{key}. {value}" for key, value in question.choices.items())
            return (
                "The following frames were retained causally before this question was known. "
                "Answer using only one option letter.\n"
                f"Question: {question.text}\n{choices}"
            )
        return (
            "The following frames were retained causally before this question was known. "
            "Answer concisely from those frames only.\n"
            f"Question: {question.text}"
        )

    def _prepare(self, question: Question, frames: Sequence[FrameObservation]):
        from qwen_vl_utils import process_vision_info

        content: list[dict[str, object]] = []
        for frame in frames:
            image_item: dict[str, object] = {"type": "image", "image": Image.fromarray(frame.image)}
            if self.min_pixels is not None:
                image_item["min_pixels"] = self.min_pixels
            if self.max_pixels is not None:
                image_item["max_pixels"] = self.max_pixels
            content.append(image_item)
            content.append({"type": "text", "text": f"Retained timestamp: {frame.ref.timestamp_sec:.3f}s"})
        content.append({"type": "text", "text": self._prompt(question)})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        return inputs

    def predict(self, question: Question, frames: Sequence[FrameObservation], **_: object) -> Prediction:
        inputs = self._prepare(question, frames)
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        scored = score_answer(question, answer)
        return Prediction(answer, str(scored["parsed_answer"]), float(scored["score"]), str(scored["metric"]))

    def correct_answer_logprob(self, question: Question, frames: Sequence[FrameObservation]) -> float:
        """Teacher-forced next-token log-probability for a closed answer letter."""
        if question.answer_type != "multiple_choice":
            raise ValueError("counterfactual log-probability requires a multiple-choice question")
        inputs = self._prepare(question, frames)
        answer_ids = self.processor.tokenizer.encode(question.answer, add_special_tokens=False)
        if not answer_ids:
            raise ValueError("correct answer produced no tokenizer ids")
        total = 0.0
        current_ids = inputs.input_ids
        current_mask = inputs.attention_mask
        extra = {key: value for key, value in inputs.items() if key not in {"input_ids", "attention_mask"}}
        with self._torch.inference_mode():
            for token_id in answer_ids:
                output = self.model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    logits_to_keep=1,
                    **extra,
                )
                log_probs = self._torch.log_softmax(output.logits[:, -1, :].float(), dim=-1)
                total += float(log_probs[0, token_id].item())
                token = self._torch.tensor([[token_id]], device=current_ids.device, dtype=current_ids.dtype)
                current_ids = self._torch.cat([current_ids, token], dim=1)
                current_mask = self._torch.cat([current_mask, self._torch.ones_like(token)], dim=1)
        return total
