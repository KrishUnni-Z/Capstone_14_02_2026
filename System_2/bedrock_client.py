"""
bedrock_client.py  shared Bedrock Converse API wrapper.

Used by llm_scoring, score_goal, verify_goal, infer_dependencies.
One call shape for anthropic, Mistral Large 2407, Llama 3.x.

Why Converse and not invoke_model:
  Converse gives a unified request/response shape across vendors. Without it
  we would need three vendor-specific body builders and three response parsers.
  Converse hides all of that.

Changing a model:
  Edit MODEL_IDS below. One line, one place.
"""

import os
import re
import json

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Disable botocore's built-in retry so we don't stack delays. We have a retry
# loop in verify_goal._call_verifier that handles throttles explicitly with
# known backoff timing; letting botocore silently retry on top added hidden
# 20s+ waits per call.
_BOTO_CONFIG = Config(retries={"max_attempts": 1, "mode": "standard"})

# Single shared client. Region from env, credentials from boto3 default chain.
_CLIENT = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-west-2"),
    config=_BOTO_CONFIG,
)


# Central source of truth for model IDs. Roles map to Bedrock model strings.
# Swap a model by editing one line here.
MODEL_IDS = {
    "llama_tester"       : "us.meta.llama3-3-70b-instruct-v1:0",
    "mistral_tester"     : "mistral.mistral-large-2407-v1:0",
    "verifier_primary"   : "us.anthropic.claude-3-haiku-20240307-v1:0",
    "verifier_fallback"  : "us.meta.llama3-1-70b-instruct-v1:0",
    "dependency_inferrer": "us.anthropic.claude-3-haiku-20240307-v1:0",
}


class BedrockCallError(Exception):
    """Raised when a Bedrock call fails after retries are exhausted."""


def call_model(role, prompt, max_tokens=1200, temperature=0.2, system=None, prefill=None):
    """
    Invoke a Bedrock model via Converse API.

    Args:
      role        : key into MODEL_IDS (e.g. "llama_tester", "verifier_primary")
      prompt      : user-turn text
      max_tokens  : maxTokens in inferenceConfig
      temperature : temperature in inferenceConfig
      system      : optional system prompt text
      prefill     : optional assistant-turn prefill (e.g. "{" to force JSON).
                    The returned text will have prefill prepended so parsers
                    see the complete response. Must not end with whitespace.

    Returns:
      str of generated text (already stripped, with prefill prepended if set)

    Raises:
      BedrockCallError on ClientError or malformed response
    """
    if role not in MODEL_IDS:
        raise ValueError(f"Unknown model role: {role!r}. Options: {list(MODEL_IDS)}")

    model_id = MODEL_IDS[role]
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    if prefill is not None:
        # Bedrock Converse rejects trailing whitespace on assistant prefill
        cleaned_prefill = prefill.rstrip()
        if cleaned_prefill:
            messages.append({"role": "assistant", "content": [{"text": cleaned_prefill}]})

    kwargs = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {
            "temperature": float(temperature),
            "maxTokens": int(max_tokens),
        },
    }
    if system:
        kwargs["system"] = [{"text": system}]

    try:
        resp = _CLIENT.converse(**kwargs)
    except ClientError as e:
        err = e.response.get("Error", {})
        raise BedrockCallError(f"{err.get('Code', 'Unknown')}: {err.get('Message', str(e))}") from e
    except Exception as e:
        raise BedrockCallError(f"{type(e).__name__}: {e}") from e

    # Defensive response parsing: find the first content block that actually has text.
    try:
        content_blocks = resp["output"]["message"]["content"]
    except (KeyError, TypeError) as e:
        raise BedrockCallError(f"Malformed Converse response: {e}") from e

    for block in content_blocks:
        if isinstance(block, dict) and "text" in block and block["text"]:
            generated = block["text"].strip()
            # Prepend the prefill if any so the caller sees complete output
            if prefill is not None:
                cleaned_prefill = prefill.rstrip()
                if cleaned_prefill and not generated.startswith(cleaned_prefill):
                    generated = cleaned_prefill + generated
            return generated

    raise BedrockCallError(f"No text content in response: {resp!r}")


def extract_json(text, required_keys=None):
    """
    Strip vendor-specific preambles and extract the first JSON-like object.

    Handles:
      <think>...</think> blocks (anthropic, some Llama prompts)
      <|thinking|>...<|/thinking|> blocks
      [INST]...[/INST] tags (Mistral)
      ```json ... ``` fenced code blocks
      Leading commentary before the JSON
      Python-style single-quoted dicts (anthropic sometimes emits these)

    Args:
      text          : raw model output
      required_keys : optional list of keys that must be present in parsed dict

    Returns:
      parsed dict

    Raises:
      ValueError if no dict-like object found or required keys missing
    """
    if not text:
        raise ValueError("Empty response")

    cleaned = text
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\[INST\].*?\[/INST\]", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```json", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No dict-like object in response: {cleaned[:250]}")
    span = match.group(0)

    parsed = None
    # Try strict JSON first (fastest, covers the clean case)
    try:
        parsed = json.loads(span)
    except json.JSONDecodeError:
        pass

    # Fallback: ast.literal_eval handles single quotes and Python dict literals,
    # which anthropic on Bedrock occasionally emits even with a JSON prefill.
    if parsed is None:
        try:
            import ast as _ast
            candidate = _ast.literal_eval(span)
            if isinstance(candidate, dict):
                parsed = candidate
        except (ValueError, SyntaxError):
            pass

    if parsed is None:
        raise ValueError(f"Could not parse dict from response: {span[:250]}")

    if required_keys:
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

    return parsed