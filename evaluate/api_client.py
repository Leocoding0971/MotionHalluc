"""One non-streaming OpenAI-compatible Chat Completions request per attempt."""

from dataclasses import dataclass

import requests

from evaluate.results import redact


class APIError(RuntimeError):
    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata


@dataclass
class Reply:
    text: str
    metadata: dict


class APIClient:
    def __init__(self, api_url, api_key, model_name, timeout=600, max_tokens=3000):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(self, content):
        metadata = {}
        try:
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model_name, "messages": [{"role": "user", "content": content}],
                      "max_tokens": self.max_tokens},
                timeout=self.timeout,
            )
            metadata = {"http_status": response.status_code,
                        "response_body": redact(response.text, self.api_key)[:16384]}
            if response.status_code >= 400:
                raise ValueError(f"HTTP {response.status_code}: {metadata['response_body'][:1000]}")
            document = response.json()
            choice = document["choices"][0]
            metadata["finish_reason"] = choice.get("finish_reason")
            value = choice["message"]["content"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("API returned no nonempty text content")
            return Reply(value, metadata)
        except Exception as error:
            message = redact(f"{type(error).__name__}: {error}", self.api_key)
            raise APIError(message, metadata) from None
