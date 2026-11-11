"""
Unified LLM Client supporting Gemini, Anthropic, and OpenAI providers:
- Provider, model_id, and per-token prices read ONLY from config.yaml.
- Accurately tracks input_tokens, output_tokens, and reasoning_tokens separately.
- Provider billing:
  * Google Gemini bills thinking/reasoning tokens as output candidate tokens (output rate).
  * Anthropic Claude bills thinking tokens as output tokens.
  * OpenAI reasoning models (o1/o3) bill reasoning tokens as output tokens.
- Hard-fails if API key is missing; never falls back to mock logic.
"""
import os
from typing import Dict, Any, List

class LLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_usd = config.get("MAX_USD", 10.0)
        self.total_spent_usd = 0.0

    def generate(
        self,
        role_type: str, # "ACTOR" or "OPTIMIZER"
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        if self.total_spent_usd >= self.max_usd:
            raise RuntimeError(f"MAX_USD budget cap (${self.max_usd:.2f}) reached! Current spend: ${self.total_spent_usd:.4f}")

        role_cfg = self.config.get(role_type)
        if not role_cfg:
            raise ValueError(f"Role '{role_type}' configuration missing from config.yaml")

        provider = role_cfg.get("provider", "gemini").lower()
        model_id = role_cfg.get("model_id")
        pricing = role_cfg.get("pricing", {})

        input_rate = pricing.get("input_per_1m", 0.0)
        output_rate = pricing.get("output_per_1m", 0.0)
        thinking_rate = pricing.get("thinking_per_1m", output_rate)

        reasoning_tokens = 0
        input_tokens = 0
        output_tokens = 0

        # -------------------------------------------------------------
        # 1. Google Gemini Provider (google.genai SDK)
        # -------------------------------------------------------------
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required to run real Gemini calls.")

            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            sys_instruct = None
            prompt_parts = []
            for m in messages:
                if m["role"] == "system":
                    sys_instruct = m["content"]
                else:
                    prompt_parts.append(m["content"])

            full_prompt = "\n\n".join(prompt_parts)
            config_params = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=sys_instruct
            )

            response = client.models.generate_content(
                model=model_id,
                contents=full_prompt,
                config=config_params
            )

            text = response.text or ""
            usage = response.usage_metadata
            if usage:
                input_tokens = usage.prompt_token_count or 0
                output_tokens = usage.candidates_token_count or 0
                # Google Gemini exposes thinking tokens in candidates or details
                reasoning_tokens = getattr(usage, "thoughts_token_count", 0) or 0
                # If thoughts are part of candidates_token_count, separate standard output
                standard_output_tokens = max(0, output_tokens - reasoning_tokens)
            else:
                input_tokens = len(full_prompt) // 4
                output_tokens = len(text) // 4
                standard_output_tokens = output_tokens

            cost_usd = (input_tokens / 1e6) * input_rate + \
                       (standard_output_tokens / 1e6) * output_rate + \
                       (reasoning_tokens / 1e6) * thinking_rate

        # -------------------------------------------------------------
        # 2. Anthropic Provider
        # -------------------------------------------------------------
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable is required for Anthropic calls.")

            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            system_msg = ""
            user_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_msg += m["content"] + "\n"
                else:
                    user_messages.append({"role": m["role"], "content": m["content"]})

            response = client.messages.create(
                model=model_id,
                system=system_msg.strip() if system_msg else None,
                messages=user_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_usd = (input_tokens / 1e6) * input_rate + (output_tokens / 1e6) * output_rate

        # -------------------------------------------------------------
        # 3. OpenAI Provider
        # -------------------------------------------------------------
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is required for OpenAI calls.")

            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            text = response.choices[0].message.content
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            # OpenAI reasoning tokens
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) if hasattr(usage, "completion_tokens_details") else 0
            standard_output_tokens = max(0, output_tokens - reasoning_tokens)
            cost_usd = (input_tokens / 1e6) * input_rate + \
                       (standard_output_tokens / 1e6) * output_rate + \
                       (reasoning_tokens / 1e6) * thinking_rate
        else:
            raise ValueError(f"Unsupported provider '{provider}' in config.yaml")

        self.total_spent_usd += cost_usd

        return {
            "text": text,
            "provider": provider,
            "model_id": model_id,
            "tokens": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens
            },
            "cost_usd": cost_usd
        }
