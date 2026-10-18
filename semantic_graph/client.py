"""Minimal DeepSeek HTTP client. Credentials never enter returned metadata."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass


class APIError(RuntimeError):
    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


@dataclass(frozen=True)
class ModelConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout: float = 90.0
    retries: int = 2

    def __post_init__(self):
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTPS URL without credentials, query, or fragment")
        if not self.model.strip() or self.max_tokens <= 0 or self.timeout <= 0 or self.retries < 0:
            raise ValueError("invalid model configuration")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    def public(self):
        return {**asdict(self), "thinking": "disabled", "response_format": "json_object"}

    def request_identity(self):
        # Changing network retry/timeout settings must not repeat paid inference.
        return {k: v for k, v in {**self.public(), "temperature": float(self.temperature)}.items()
                if k not in {"timeout", "retries"}}


class DeepSeekClient:
    def __init__(self, config: ModelConfig, api_key=None):
        self.config = config
        self._key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self._key:
            raise ValueError("DEEPSEEK_API_KEY is not configured (or use --key-stdin)")

    def _request(self, suffix, payload=None):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + suffix, data=data,
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json",
                     "User-Agent": "VLM-Jailbreak-semantic-graph/1"},
            method="POST" if payload is not None else "GET",
        )
        for attempt in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                code = exc.code
                # Do not echo response bodies or headers (which could contain secrets).
                exc.close()
                retryable = code == 429 or code >= 500
                if not retryable or attempt == self.config.retries:
                    raise APIError(f"DeepSeek HTTP {code}", fatal=code in {400, 401, 402, 403, 404}) from None
            except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, json.JSONDecodeError):
                if attempt == self.config.retries:
                    raise APIError("DeepSeek network/timeout/response-decode error") from None
            time.sleep(min(2 ** attempt, 4))
        raise AssertionError("unreachable")

    def models(self):
        data = self._request("/models")
        return [item["id"] for item in data.get("data", []) if isinstance(item, dict) and "id" in item]

    def complete(self, messages):
        raw = self._request("/chat/completions", {
            "model": self.config.model, "messages": messages,
            "response_format": {"type": "json_object"}, "temperature": self.config.temperature,
            "thinking": {"type": "disabled"}, "max_tokens": self.config.max_tokens, "stream": False,
        })
        try:
            choice = raw["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise TypeError("unsupported content")
            return {
                "content": content or "", "refusal": bool(message.get("refusal")),
                "finish_reason": choice.get("finish_reason"), "model": raw.get("model"),
                "response_id": raw.get("id"), "usage": raw.get("usage", {}),
            }
        except (KeyError, IndexError, TypeError, AttributeError):
            raise APIError("DeepSeek returned an unsupported response envelope") from None

