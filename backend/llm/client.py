"""Local Qwen 0.6B client built on top of Hugging Face transformers."""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, Iterator, Optional

from concurrent.futures import Future, ThreadPoolExecutor

import torch
from transformers import TextIteratorStreamer, pipeline

_CLIENT: Optional["LocalQwenClient"] = None
_CLIENT_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-loader")
_LOADING_FUTURE: Optional[Future] = None


class LocalQwenClient:
    """Lazy-loading wrapper around the Qwen3 0.6B chat pipeline."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
        temperature: float = 0.3,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or 25600
        self.temperature = temperature
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self._pipeline_kwargs = self._build_pipeline_kwargs(device)

    def _load_pipeline(self):
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    kwargs: Dict[str, Any] = {
                        "model": self.model_name,
                        "task": "text-generation",
                        "trust_remote_code": True,
                    }
                    kwargs.update(self._pipeline_kwargs)

                    self._pipeline = pipeline(**kwargs)
        return self._pipeline

    def _build_pipeline_kwargs(self, requested_device: Optional[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}

        device_choice = self._parse_device(requested_device)
        if device_choice is None:
            device_choice = self._auto_device()

        if isinstance(device_choice, int):
            kwargs["device"] = device_choice
        elif isinstance(device_choice, str):
            lowered = device_choice.lower()
            if lowered in {"cpu", "cuda", "mps"} or lowered.startswith("cuda:"):
                kwargs["device"] = device_choice
            elif lowered == "auto":
                kwargs["device_map"] = "auto"
            else:
                kwargs["device_map"] = device_choice

        if kwargs.get("device"):
            lowered = str(kwargs["device"]).lower()
            if lowered.startswith("cuda"):
                kwargs["torch_dtype"] = torch.float16
            elif lowered == "mps":
                kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float32
        elif kwargs.get("device_map") == "auto" and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        return kwargs

    @staticmethod
    def _parse_device(value: Optional[str] | int | float) -> Optional[str | int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        value = value.strip()
        if value == "":
            return None
        if value.isdigit():
            return int(value)
        return value

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            # Default to CPU for broader compatibility; allow opting in via env.
            return "cpu"
        return "cpu"

    def _build_messages(self, prompt: str) -> list[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "First, think step-by-step in <think> tags. Keep it brief. "
                    "Then, on a new line, output a JSON object with only the 'action' key."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str) -> Dict[str, Any]:
        result: Optional[Dict[str, Any]] = None
        for event in self.generate_stream(prompt):
            if event.get("type") == "result":
                result = event["data"]
            elif event.get("type") == "error":
                return {
                    "thinking_process": event.get("error", "Model error"),
                    "action": None,
                }

        if result is None:
            return {
                "thinking_process": "Model did not return any data.",
                "action": None,
            }

        return result

    def generate_stream(self, prompt: str) -> Iterator[Dict[str, Any]]:
        pipe = self._load_pipeline()
        messages = self._build_messages(prompt)

        streamer = TextIteratorStreamer(
            pipe.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_error: Optional[Exception] = None

        def run_pipeline() -> None:
            nonlocal generation_error
            try:
                kwargs: Dict[str, Any] = {
                    "temperature": self.temperature,
                    "do_sample": self.temperature > 0,
                    "streamer": streamer,
                    "return_full_text": False,
                }

                max_tokens = self._effective_max_new_tokens(pipe)
                if max_tokens is not None:
                    kwargs["max_new_tokens"] = max_tokens

                pipe(messages, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive guard
                generation_error = exc

        worker = threading.Thread(target=run_pipeline, daemon=True)
        worker.start()

        collected_parts: list[str] = []

        for text in streamer:
            if not text:
                continue
            collected_parts.append(text)
            yield {"type": "token", "token": text}

        worker.join(timeout=30.0)  # Add a 30-second timeout

        if worker.is_alive():
            yield {"type": "error", "error": "Model generation timed out."}
            return

        if generation_error is not None:
            yield {"type": "error", "error": str(generation_error)}
            return

        raw_text = "".join(collected_parts)
        parsed = _parse_response(raw_text)
        parsed.setdefault("raw_response", raw_text)
        yield {"type": "result", "data": parsed}

    def _effective_max_new_tokens(self, pipe) -> Optional[int]:
        if self.max_new_tokens and self.max_new_tokens > 0:
            return self.max_new_tokens

        model = getattr(pipe, "model", None)
        config = getattr(model, "config", None)
        if config is not None:
            for attr in ("max_new_tokens", "max_length", "max_position_embeddings"):
                value = getattr(config, attr, None)
                if isinstance(value, int) and value > 0:
                    # Use a generous ceiling that still respects context limits.
                    if attr == "max_position_embeddings":
                        return max(32, value - getattr(config, "max_input_length", 0))
                    return value

        # Fallback to a conservative but high ceiling.
        return 1024


def get_qwen_action(prompt: str) -> Dict[str, Any]:
    client = _get_client()

    if not _ensure_pipeline_ready(client):
        return {
            "thinking_process": (
                "Model is still downloading/warming up. "
                "Returning a fallback action for now."
            ),
            "action": None,
        }

    return client.generate(prompt)


def stream_qwen_action(prompt: str) -> Iterator[Dict[str, Any]]:
    client = _get_client()

    if not _ensure_pipeline_ready(client):
        yield {
            "type": "result",
            "data": {
                "thinking_process": (
                    "Model is still downloading/warming up. Returning a fallback action for now."
                ),
                "action": None,
            },
        }
        return

    yield from client.generate_stream(prompt)


def _get_client() -> LocalQwenClient:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                max_new_tokens_env = os.getenv("QWEN_MAX_NEW_TOKENS")
                max_new_tokens: Optional[int]
                if max_new_tokens_env is None or max_new_tokens_env.strip().lower() in {"", "none", "unlimited"}:
                    max_new_tokens = None
                else:
                    try:
                        parsed = int(max_new_tokens_env)
                        max_new_tokens = parsed if parsed > 0 else None
                    except ValueError:
                        max_new_tokens = None

                _CLIENT = LocalQwenClient(
                    model_name=os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                    device=os.getenv("QWEN_DEVICE"),
                    max_new_tokens=max_new_tokens,
                    temperature=float(os.getenv("QWEN_TEMPERATURE", "0.3")),
                )
    return _CLIENT


def _ensure_pipeline_ready(client: LocalQwenClient) -> bool:
    global _LOADING_FUTURE

    if client._pipeline is not None:
        return True

    if _LOADING_FUTURE is None:
        _LOADING_FUTURE = _EXECUTOR.submit(client._load_pipeline)

    if _LOADING_FUTURE.done():
        _LOADING_FUTURE.result()  # re-raise if failed
        return True

    return False


def _extract_generated_text(outputs: Any) -> str:
    if not outputs:
        return ""
    result = outputs[0]
    generated = result.get("generated_text") if isinstance(result, dict) else None

    if isinstance(generated, str):
        return generated

    if isinstance(generated, list):
        # New chat pipeline returns a list of message dicts
        assistant_messages = [m for m in generated if m.get("role") == "assistant"]
        if assistant_messages:
            last = assistant_messages[-1]
            content = last.get("content")
            if isinstance(content, list):
                return "\n".join(
                    part.get("text", "") for part in content if part.get("type") == "text"
                ).strip()
            if isinstance(content, str):
                return content.strip()
        # Fallback to concatenating
        return "\n".join(str(item) for item in generated)

    for key in ("text", "content"):
        if isinstance(result, dict) and key in result:
            return str(result[key])

    return str(result)


JSON_BLOCK_PATTERN = re.compile(r"\{[\s\S]*\}")
THINK_BLOCK_PATTERN = re.compile(r"<think>([\s\S]*?)</think>")


def _parse_response(text: str) -> Dict[str, Any]:
    json_match = JSON_BLOCK_PATTERN.search(text)
    if not json_match:
        return {"thinking_process": text.strip(), "action": None}

    json_str = json_match.group(0)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback for malformed JSON, but still try to get thoughts
        think_match = THINK_BLOCK_PATTERN.search(text)
        thinking = think_match.group(1).strip() if think_match else text.strip()
        return {"thinking_process": thinking, "action": None}

    action = data.get("action")

    think_match = THINK_BLOCK_PATTERN.search(text)
    thinking = think_match.group(1).strip() if think_match else ""

    # If thinking is in the JSON, prefer that, otherwise use the <think> block
    thinking_in_json = data.get("thinking_process") or data.get("thoughts")
    if thinking_in_json:
        thinking = str(thinking_in_json)

    return {"thinking_process": thinking, "action": action}


"""
Optional import-time warm-up
By default we avoid auto-warmup at import to prevent unexpected heavy
initialization (which can crash constrained environments like Colab).
Opt-in by setting environment variable QWEN_AUTOWARMUP to a truthy value
('1', 'true', 'yes').
"""
import os as _os  # local alias to avoid polluting public namespace further

_AUTO_WARMUP_FLAG = str(_os.getenv("QWEN_AUTOWARMUP", "0")).strip().lower()
_AUTO_WARMUP_ENABLED = _AUTO_WARMUP_FLAG in {"1", "true", "yes", "on"}

if _AUTO_WARMUP_ENABLED:
    try:  # Fire-and-forget warm-up so downloads start early
        _ensure_pipeline_ready(_get_client())
    except Exception:  # pragma: no cover - best-effort warm-up
        pass