"""One OpenAI-compatible request per attempt, without implicit SDK retries."""

import requests


class APIError(RuntimeError):
    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata


class APIClient:
    def __init__(self, settings):
        self.settings = settings

    def complete(self, messages):
        cfg = self.settings
        body = {"model": cfg["model_name"], "messages": messages}
        if cfg["max_tokens"] is not None:
            body["max_tokens"] = cfg["max_tokens"]
        metadata = {}
        try:
            response = requests.post(cfg["api_url"], headers={"Authorization": f"Bearer {cfg['api_key']}",
                                     "Content-Type": "application/json"}, json=body, timeout=cfg["timeout"])
            metadata = {"http_status": response.status_code, "response_body": response.text}
            if response.status_code >= 400:
                raise ValueError(f"HTTP {response.status_code}")
            document = response.json()
            choice = document["choices"][0]
            metadata["finish_reason"] = choice.get("finish_reason")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("API returned no nonempty text content")
            return content, metadata
        except Exception as error:
            raise APIError(f"{type(error).__name__}: {error}", metadata) from None
