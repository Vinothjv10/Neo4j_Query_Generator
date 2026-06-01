import os
import json
import time

from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx

from app.config import settings
from app.utils.logger import log_step

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
# Refresh token 60 seconds before it expires (tokens last 3600s)
_TOKEN_REFRESH_BUFFER_SECS = 300


class LLMService:
    def __init__(self):
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._credentials = None

    async def _ensure_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expiry - _TOKEN_REFRESH_BUFFER_SECS:
            return self._token

        creds_path = settings.google_application_credentials
        if not creds_path or not os.path.exists(creds_path):
            raise FileNotFoundError(
                f"Service account file not found: {creds_path}. "
                "Set GOOGLE_APPLICATION_CREDENTIALS in .env"
            )

        abs_path = os.path.abspath(creds_path)
        self._credentials = service_account.Credentials.from_service_account_file(
            abs_path,
            scopes=SCOPES,
        )

        auth_req = GoogleAuthRequest()
        self._credentials.refresh(auth_req)
        self._token = self._credentials.token
        # google-auth sets expiry on the credentials object
        expiry = getattr(self._credentials, "expiry", None)
        if expiry is not None:
            import datetime
            self._token_expiry = expiry.timestamp() if hasattr(expiry, "timestamp") else (now + 3600)
        else:
            self._token_expiry = now + 3600

        project = settings.vertex_ai_project
        if not project:
            with open(abs_path) as f:
                project = json.load(f).get("project_id", "")

        log_step("GCP", "Vertex AI credentials loaded/refreshed",
                 project=project,
                 location=settings.vertex_ai_location,
                 model=settings.vertex_ai_model)
        return self._token

    def _make_url(self) -> str:
        project = settings.vertex_ai_project
        location = settings.vertex_ai_location
        model = settings.vertex_ai_model
        return (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project}/locations/{location}/"
            f"publishers/anthropic/models/{model}:rawPredict"
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict:
        """
        Call Claude with tool_use support. Returns the raw response dict
        containing {content: [...], stop_reason: str}.
        """
        token = await self._ensure_token()
        url = self._make_url()

        payload = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": messages,
            "system": system_prompt,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        log_step("LLM", "Calling Claude with tools",
                 tools=[t["name"] for t in tools],
                 messages=len(messages))

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise ValueError(
                f"Vertex AI API error (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )

        data = response.json()
        return {
            "content": data.get("content", []),
            "stop_reason": data.get("stop_reason", "end_turn"),
        }

    async def generate_sql(self, system_prompt: str, user_prompt: str) -> str:
        token = await self._ensure_token()
        url = self._make_url()

        log_step("LLM", "Calling Claude on Vertex AI",
                 model=settings.vertex_ai_model, max_tokens=2048, temperature=0.10)

        payload = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "system": system_prompt,
            "max_tokens": 2048,
            "temperature": 0.10,
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise ValueError(
                f"Vertex AI API error (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )

        data = response.json()
        content_blocks = data.get("content", [])
        raw = ""
        for block in content_blocks:
            if block.get("type") == "text":
                raw += block.get("text", "")

        raw = raw.strip()
        log_step("LLM", f"Raw response received", chars=len(raw))

        if raw.startswith("UNABLE_TO_GENERATE"):
            raise ValueError(
                "The LLM could not generate a SQL query for this question "
                "with the available schema."
            )

        sql = self._clean_sql(raw)
        log_step("LLM", f"SQL after cleaning", sql=sql.replace("\n", " "))

        if not sql:
            raise ValueError("LLM returned an empty or invalid response.")

        return sql

    def _clean_sql(self, raw: str) -> str:
        if raw.startswith("```"):
            lines = raw.splitlines()
            cleaned: list[str] = []
            for line in lines:
                if line.startswith("```"):
                    continue
                cleaned.append(line)
            return "\n".join(cleaned).strip()
        return raw
