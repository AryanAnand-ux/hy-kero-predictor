"""
ai_provider.py
──────────────
Abstracts the LLM interaction. Implements direct HTTP integration with 
Google Gemini API and falls back to a smart, domain-aware mock provider 
if no API key is configured.
"""

import os
import json
import httpx
import logging
import asyncio
from typing import AsyncGenerator

try:
    from .constants import (
        DEFAULT_MODEL_NAME,
        DEFAULT_TEMPERATURE,
        DEFAULT_MAX_OUTPUT_TOKENS,
        DEFAULT_TIMEOUT_SECONDS,
    )
except ImportError:
    from constants import (
        DEFAULT_MODEL_NAME,
        DEFAULT_TEMPERATURE,
        DEFAULT_MAX_OUTPUT_TOKENS,
        DEFAULT_TIMEOUT_SECONDS,
    )

logger = logging.getLogger("hykero.ai")


class AIProvider:
    """Base abstraction for AI chat generation."""
    
    def generate_response(self, prompt: str, history: list) -> str:
        """Generate a complete text response."""
        raise NotImplementedError()

    async def generate_response_stream(self, prompt: str, history: list) -> AsyncGenerator[str, None]:
        """Stream chunks of text response."""
        raise NotImplementedError()
        yield ""


class GeminiAIProvider(AIProvider):
    """Google Gemini API Provider using standard REST endpoints."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = DEFAULT_MODEL_NAME
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
        self.url = f"{self.base_url}:generateContent"
        self.stream_url = f"{self.base_url}:streamGenerateContent"
        self.headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        self.system_instruction = (
            "You are an AI assistant specialized in heavy kerosene processing, CDU operations, "
            "and machine learning models. You are helping engineers at the IOCL Crude Distillation Unit. "
            "The current ML model is a Huber Tuned regressor predicting "
            "Flash Point with a test RMSE of 2.21°C and R2 of 0.6708. We use a 75-minute process lag alignment "
            "and completely removed target lags (lag1_flash_gc) to prevent data leakage. Always provide detailed, professional "
            "answers. Keep responses relatively concise but technically accurate."
        )

    def _build_payload(self, prompt: str, history: list) -> dict:
        contents = []
        for msg in history:
            role_map = {"user": "user", "assistant": "model"}
            contents.append({
                "role": role_map.get(msg["role"], "user"),
                "parts": [{"text": msg["content"]}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        return {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": self.system_instruction}]
            },
            "generationConfig": {
                "temperature": DEFAULT_TEMPERATURE,
                "maxOutputTokens": DEFAULT_MAX_OUTPUT_TOKENS
            }
        }

    def generate_response(self, prompt: str, history: list) -> str:
        payload = self._build_payload(prompt, history)
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
                r = client.post(self.url, json=payload, headers=self.headers)
                r.raise_for_status()
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except Exception as e:
            logger.warning(f"Gemini API error (generate_response), falling back: {e}")
            fallback_resp = get_mock_ai_provider().generate_response(prompt, history)
            return f"*(Gemini API returned rate limit/error. Falling back to offline assistant)*\n\n{fallback_resp}"

    async def generate_response_stream(self, prompt: str, history: list) -> AsyncGenerator[str, None]:
        payload = self._build_payload(prompt, history)
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
                async with client.stream("POST", self.stream_url, json=payload, headers=self.headers) as response:
                    response.raise_for_status()
                    buffer = ""
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        buffer += line
                        try:
                            clean_buf = buffer.strip().lstrip("[").rstrip("]").rstrip(",")
                            if clean_buf.startswith("{") and clean_buf.endswith("}"):
                                chunk_data = json.loads(clean_buf)
                                text = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                                yield text
                                buffer = ""
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Gemini API error (generate_response_stream), falling back: {e}")
            yield "*(Gemini API returned rate limit/error. Falling back to offline assistant)*\n\n"
            async for chunk in get_mock_ai_provider().generate_response_stream(prompt, history):
                yield chunk


class MockAIProvider(AIProvider):
    """Smart domain-specific Mock AI Provider running locally."""

    def __init__(self):
        self.responses = [
            {
                "keywords": ["spec", "limit", "range", "normal"],
                "text": (
                    "The normal operating specification range for Heavy Kerosene (HY Kero) Flash Point "
                    "at the IOCL Crude Distillation Unit (CDU) is **63.0°C to 96.0°C**. "
                    "If the Flash Point drops below 63.0°C, it indicates light-end contamination (danger of flammability). "
                    "If it exceeds 96.0°C, the product is too heavy (poor combustion properties)."
                )
            },
            {
                "keywords": ["rmse", "r2", "accuracy", "performance", "metrics"],
                "text": (
                    "The optimized prediction pipeline uses a **Tuned Huber Regressor** model. "
                    "The current model evaluation metrics on the test set are:\n"
                    "- **Test RMSE**: **2.21°C**\n"
                    "- **Test R² Score**: **0.6708** (representing stable variance capture across distribution shifts)\n"
                    "- **Test MAPE**: **2.08%**\n"
                    "This performance represents a highly accurate and generalization-optimized model operating near the lab noise floor (~1.5-2.0°C)."
                )
            },
            {
                "keywords": ["shift", "distribution"],
                "text": (
                    "We identified a statistically significant **distribution shift** (+3.71°C target mean difference) "
                    "between the training split (Apr–Jan) and the test split (Feb–Mar). To protect the model from overfitting, "
                    "our pipeline trains a robust Huber Regressor with L1 robust loss and uses standard scaling. "
                    "This allows the model to generalize and extrapolate accurately when column temperatures run hotter in 2026."
                )
            },
            {
                "keywords": ["lag", "residence", "dead time", "delay"],
                "text": (
                    "The model uses a **75-minute process residence time lag** to align the raw sensor data with the "
                    "rundown sampling times. Heavy kerosene takes approximately 75 minutes to travel from the column "
                    "draw-off tray through the stripping column and heat exchange circuit to the physical sample point. "
                    "Note: Target lags (like `lag1_flash_gc`) were completely removed from the feature set to prevent data leakage."
                )
            },
            {
                "keywords": ["steam", "ss_", "stripper", "11c5"],
                "text": (
                    "Stripping steam (measured by tags starting with `SS_`) is critical for adjusting the Flash Point. "
                    "Steam injected at the bottom of the Heavy Kerosene stripper column vaporizes light components, driving "
                    "them back up into the main column. This raises the Flash Point of the bottom product. The feature "
                    "`SS_11C5_mean` is currently the highest-ranking physical feature in feature importance."
                )
            },
            {
                "keywords": ["temp", "temperature", "mf_hk_draw", "outlet", "furnace"],
                "text": (
                    "Temperatures dictate vapor-liquid equilibrium in the CDU. Specifically:\n"
                    "- `MF_HK_Draw_T_mean` (Heavy Kerosene draw-off tray temperature) is highly positively correlated with "
                    "the flash point. Higher temperatures mean lighter fractions are kept in vapor phase, leaving a heavier bottoms product.\n"
                    "- Furnace outlet temperatures (`Outlet_temp_11F1` to `11F4`) represent heat duty and dictate total column feed vaporization."
                )
            },
            {
                "keywords": ["hello", "hi", "hey", "help", "who"],
                "text": (
                    "Hello! I am the HY Kero Process Assistant. I can answer your technical questions about the "
                    "IOCL CDU Heavy Kerosene Flash Point Predictor model, model metrics, feature engineering "
                    "(e.g., process lag, shift filtering), and the process physics (stripping steam, draw temps, column operation). "
                    "I can also look up historical Flash Point predictions for specific dates and times."
                )
            }
        ]
        self.default_response = (
            "I am running in local offline mode. I can answer questions about the HY Kero ML model metrics, "
            "feature engineering (like the 75-min process lag and leakage prevention), stripping steam effects, "
            "CDU tray temperatures, and normal operating specifications (63-96°C). "
            "I can also look up historical predictions for specific dates. "
            "For general queries, please set the `GEMINI_API_KEY` environment variable."
        )

    def _format_db_context_response(self, prompt: str) -> str | None:
        if "[DATABASE LOOKUP:" in prompt and "No prediction records found" in prompt:
            import re
            m = re.search(r'No prediction records found for ([^\.\]]+)', prompt)
            date_info = m.group(1).strip() if m else "the requested date"
            return (
                f"I looked up the prediction database for **{date_info}**, but "
                f"no Flash Point records were found for that date/time.\n\n"
                f"The database contains predictions from the CDU operational period. "
                f"The requested date may fall outside the available data range, or "
                f"no samples were collected during that time window.\n\n"
                f"**Tip:** Try asking for a date between April 2025 and March 2026, "
                f"which is the period covered by the training and test datasets."
            )
        
        if "[DATABASE LOOKUP RESULTS" in prompt:
            import re
            header_m = re.search(r'\[DATABASE LOOKUP RESULTS for ([^—]+)—\s*(\d+) record', prompt)
            date_info = header_m.group(1).strip() if header_m else "the requested date"
            num_records = header_m.group(2) if header_m else "?"
            
            records = re.findall(
                r'•\s*([^|]+)\|\s*Shift:\s*(\w+)\s*\|\s*Actual:\s*([^\|]+)\|\s*Predicted:\s*([^\|]+)\|\s*Residual:\s*([^\|]+)\|\s*95% CI:\s*([^\n]+)',
                prompt
            )
            
            if records:
                lines = [f"📊 **Flash Point Data for {date_info}** ({num_records} record(s)):\n"]
                for ts, shift, actual, predicted, residual, ci in records:
                    ts = ts.strip()
                    actual = actual.strip()
                    predicted = predicted.strip()
                    residual = residual.strip()
                    ci = ci.strip()
                    lines.append(
                        f"🕐 **{ts}** (Shift: {shift})\n"
                        f"   • Actual Flash Point: **{actual}**\n"
                        f"   • Predicted (ML Model): **{predicted}**\n"
                        f"   • Prediction Error: {residual}\n"
                        f"   • 95% Confidence Interval: {ci}\n"
                    )
                
                if len(records) == 1:
                    act_val = records[0][2].strip()
                    pred_val = records[0][3].strip()
                    lines.append(
                        f"The closest record shows an actual Flash Point of **{act_val}** "
                        f"vs. the model's prediction of **{pred_val}**."
                    )
                
                return "\n".join(lines)
        
        return None

    def generate_response(self, prompt: str, history: list) -> str:
        db_response = self._format_db_context_response(prompt)
        if db_response:
            return db_response
        
        clean_prompt = prompt.split("\n\n[DATABASE")[0].lower() if "[DATABASE" in prompt else prompt.lower()
        
        for resp in self.responses:
            if any(kw in clean_prompt for kw in resp["keywords"]):
                return resp["text"]
        return self.default_response

    async def generate_response_stream(self, prompt: str, history: list) -> AsyncGenerator[str, None]:
        response_text = self.generate_response(prompt, history)
        words = response_text.split(" ")
        for i, word in enumerate(words):
            yield (word + " " if i < len(words) - 1 else word)
            await asyncio.sleep(0.02)


_mock_ai_provider = None


def get_mock_ai_provider() -> MockAIProvider:
    global _mock_ai_provider
    if _mock_ai_provider is None:
        _mock_ai_provider = MockAIProvider()
    return _mock_ai_provider


def get_ai_provider() -> AIProvider:
    """Factory helper to retrieve the correct AI provider based on environment config."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip():
        logger.info("Instantiating GeminiAIProvider for chatbot")
        return GeminiAIProvider(api_key.strip())
    else:
        logger.info("No GEMINI_API_KEY found. Falling back to MockAIProvider")
        return get_mock_ai_provider()
