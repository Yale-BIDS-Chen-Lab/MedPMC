"""Model-family-neutral LMDeploy backend with true batched VLM inference."""

from __future__ import annotations

import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from .base import InferenceRequest, InferenceResult


class LMDeployBackend:
    """Run batched multimodal prompts through LMDeploy.

    The backend does not assume InternVL. Model-specific behavior is selected by
    the model repository and, only when necessary, ``chat_template``.
    """

    name = "lmdeploy"

    def __init__(
        self,
        model_name: str,
        *,
        engine: str = "turbomind",
        session_len: int = 32768,
        max_new_tokens: int = 2048,
        tensor_parallel: int = 1,
        chat_template: str | None = None,
        engine_max_batch_size: int | None = None,
        vision_max_batch_size: int | None = 16,
        cache_max_entry_count: float = 0.8,
        image_loader_workers: int = 8,
        trust_remote_code: bool = False,
        repetition_penalty: float = 1.0,
        repetition_ngram_size: int = 20,
        repetition_ngram_threshold: int = 3,
    ) -> None:
        try:
            from lmdeploy import (
                ChatTemplateConfig,
                GenerationConfig,
                PytorchEngineConfig,
                TurbomindEngineConfig,
                VisionConfig,
                pipeline,
            )
            from lmdeploy.vl import load_image
        except ImportError as exc:  # pragma: no cover - exercised on GPU systems
            raise RuntimeError(
                "LMDeploy is missing. Install the Stage 4 LMDeploy extra in its "
                "dedicated environment."
            ) from exc

        engine = str(engine).lower()
        if engine not in {"turbomind", "pytorch"}:
            raise ValueError("engine must be 'turbomind' or 'pytorch'")
        self.engine = engine
        self.model_name = model_name
        self.chat_template = chat_template or "auto"
        self.image_loader_workers = max(1, int(image_loader_workers))
        self._load_image = load_image

        config_cls = (
            TurbomindEngineConfig if engine == "turbomind" else PytorchEngineConfig
        )
        candidate_kwargs: dict[str, Any] = {
            "session_len": int(session_len),
            "tp": int(tensor_parallel),
            "max_batch_size": (
                int(engine_max_batch_size)
                if engine_max_batch_size is not None
                else None
            ),
            "cache_max_entry_count": float(cache_max_entry_count),
        }
        backend_config = config_cls(
            **_supported_kwargs(config_cls, candidate_kwargs)
        )

        pipeline_kwargs: dict[str, Any] = {
            "backend_config": backend_config,
            "trust_remote_code": bool(trust_remote_code),
        }
        if chat_template:
            pipeline_kwargs["chat_template_config"] = ChatTemplateConfig(
                model_name=chat_template
            )
        if vision_max_batch_size is not None:
            pipeline_kwargs["vision_config"] = VisionConfig(
                max_batch_size=int(vision_max_batch_size)
            )

        try:
            self._pipe = pipeline(model_name, **pipeline_kwargs)
        except AttributeError as exc:
            if "llm_config" in str(exc) and not trust_remote_code:
                raise RuntimeError(
                    "This InternVL-style repository requires Hugging Face custom "
                    "configuration code. Re-run with --trust-remote-code, or use "
                    "the pipeline's auto-trusted released MedPMC checkpoint."
                ) from exc
            raise
        self._GenerationConfig = GenerationConfig
        self._default_max_new_tokens = int(max_new_tokens)
        self._generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "temperature": 0.0,
            "repetition_penalty": float(repetition_penalty),
        }
        # LMDeploy's repeated-ngram early stopping is implemented by the
        # PyTorch engine. Do not advertise it as effective for TurboMind.
        if engine == "pytorch":
            self._generation_kwargs.update(
                repetition_ngram_size=int(repetition_ngram_size),
                repetition_ngram_threshold=int(repetition_ngram_threshold),
            )

    def _load_images(self, paths: Sequence[Path]) -> list[Any]:
        values = [str(path) for path in paths]
        if self.image_loader_workers == 1 or len(values) <= 1:
            return [self._load_image(value) for value in values]
        with ThreadPoolExecutor(max_workers=self.image_loader_workers) as executor:
            return list(executor.map(self._load_image, values))

    def generate_batch(
        self,
        requests: Sequence[InferenceRequest],
        *,
        max_new_tokens: int | None = None,
    ) -> list[InferenceResult]:
        if not requests:
            return []

        started = time.perf_counter()
        image_load_started = time.perf_counter()
        flat_paths = [path for request in requests for path in request.image_paths]
        loaded = self._load_images(flat_paths)
        image_load_seconds = time.perf_counter() - image_load_started

        prompts = []
        cursor = 0
        for request in requests:
            request_images = loaded[cursor : cursor + request.image_count]
            cursor += request.image_count
            prompts.append((request.prompt, request_images))

        generation_budget = int(
            max_new_tokens
            if max_new_tokens is not None
            else self._default_max_new_tokens
        )
        generation_config = self._GenerationConfig(
            max_new_tokens=generation_budget,
            **self._generation_kwargs,
        )
        generation_started = time.perf_counter()
        responses = self._pipe(
            prompts,
            gen_config=generation_config,
            use_tqdm=False,
        )
        generation_seconds = time.perf_counter() - generation_started
        if not isinstance(responses, list):
            responses = [responses]
        if len(responses) != len(requests):
            raise RuntimeError(
                f"LMDeploy returned {len(responses)} responses for "
                f"{len(requests)} requests"
            )

        elapsed = time.perf_counter() - started
        shared_metadata = {
            "batch_size": len(requests),
            "batch_image_count": sum(request.image_count for request in requests),
            "batch_seconds": elapsed,
            "image_load_seconds": image_load_seconds,
            "generation_seconds": generation_seconds,
            "max_new_tokens": generation_budget,
        }
        outputs = []
        for request, response in zip(requests, responses, strict=True):
            metadata = {
                **shared_metadata,
                "finish_reason": str(getattr(response, "finish_reason", "") or ""),
                "input_token_len": int(getattr(response, "input_token_len", 0) or 0),
                "generate_token_len": int(getattr(response, "generate_token_len", 0) or 0),
            }
            outputs.append(
                InferenceResult(
                    request_id=request.request_id,
                    text=str(getattr(response, "text", "") or ""),
                    metadata=metadata,
                )
            )
        return outputs

    def close(self) -> None:
        close = getattr(self._pipe, "close", None)
        if callable(close):
            close()


def _supported_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Filter optional config arguments for LMDeploy API-version tolerance."""
    parameters = inspect.signature(callable_obj).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    return {
        key: value
        for key, value in values.items()
        if value is not None and (accepts_kwargs or key in parameters)
    }
