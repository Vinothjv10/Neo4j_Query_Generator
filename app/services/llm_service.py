import os
import json

import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
from google.oauth2 import service_account

from app.config import settings
from app.utils.logger import log_step


class LLMService:
    def __init__(self):
        self._client_initialized = False

    async def _ensure_client(self):
        if self._client_initialized:
            return

        creds_path = settings.google_application_credentials
        if not creds_path or not os.path.exists(creds_path):
            raise FileNotFoundError(
                f"Service account file not found: {creds_path}. "
                "Set GOOGLE_APPLICATION_CREDENTIALS in .env"
            )

        abs_path = os.path.abspath(creds_path)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path

        credentials = service_account.Credentials.from_service_account_file(
            abs_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        project = settings.vertex_ai_project
        if not project:
            with open(abs_path) as f:
                project = json.load(f).get("project_id", "")
                log_step("GCP", f"Auto-detected project from service account", project=project)

        vertexai.init(
            project=project,
            location=settings.vertex_ai_location,
            credentials=credentials,
        )

        self._client_initialized = True
        log_step("GCP", f"Vertex AI initialized", project=project, location=settings.vertex_ai_location)

    async def generate_sql(self, system_prompt: str, user_prompt: str) -> str:
        await self._ensure_client()

        model_name = settings.vertex_ai_model
        log_step("LLM", f"Calling {model_name}", max_tokens=2048, temperature=0.10)

        safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]

        model = GenerativeModel(
            model_name=model_name,
            system_instruction=[system_prompt],
            safety_settings=safety_settings,
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.10,
                "top_p": 0.60,
            },
        )

        response = await model.generate_content_async(user_prompt)

        raw = response.text.strip()
        log_step("LLM", f"Raw response received", chars=len(raw))

        if raw == "UNABLE_TO_GENERATE":
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
