import requests
from swe_pro.inference.llm_client.base import BaseLLMClient

class QwenNIMClient(BaseLLMClient):
    """Client for Qwen models on NVIDIA NIM (e.g. qwen/qwen3.5-397b-a17b).

    Uses raw requests with enable_thinking=False in chat_template_kwargs.
    """

    ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
    ):
        super().__init__(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.api_key = api_key
        self.thinking = thinking

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }

        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        response = requests.post(self.ENDPOINT, headers=headers, json=payload, timeout=600)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
