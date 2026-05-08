"""Claude Opus-powered credit risk report generation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

_SYSTEM_PROMPT = (
    "You are a senior credit risk analyst at a Taiwanese bank. "
    "Generate concise, professional credit assessment reports in Traditional Chinese (zh-TW). "
    "Use markdown formatting. Be specific about risk drivers and avoid generic language."
)

_SECTION_INSTRUCTION = (
    "Output markdown with exactly these four sections:\n"
    "## 借款人摘要\n"
    "## 風險評分與等級\n"
    "## 主要風險因子（前 3 名 SHAP 特徵的白話解釋）\n"
    "## 建議行動\n\n"
    "Reference the specific feature values and SHAP scores provided. "
    "Do not use generic filler language."
)


def generate_credit_report(
    borrower_data: dict[str, Any],
    prediction: dict[str, Any],
    shap_top_features: list[tuple[str, float, Any]],
) -> str:
    """Generate a Traditional Chinese credit assessment report via Claude Opus 4.7.

    Args:
        borrower_data:     Borrower feature values used for credit risk evaluation.
        prediction:        Prediction output containing risk_score and risk_level.
        shap_top_features: Top SHAP feature tuples as (feature_name, shap_value, raw_value).

    Returns:
        Markdown credit assessment report, or a fallback failure message.
    """
    from anthropic import Anthropic
    from dotenv import load_dotenv

    load_dotenv()

    score = prediction.get("risk_score", "N/A")
    top_3_features = [feature[0] for feature in shap_top_features[:3]]
    fallback = f"[報告生成失敗] 風險評分: {score} | 主要因子: {top_3_features}"

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return fallback

    client = Anthropic(api_key=api_key)

    # Build a structured user prompt grounding Claude in the computed evidence.
    # All SHAP values and raw feature values are injected here so the model
    # cannot introduce claims that are not backed by the pipeline's output.
    shap_payload = [
        {"feature_name": name, "shap_value": round(float(value), 4), "raw_value": raw_value}
        for name, value, raw_value in shap_top_features
    ]
    user_prompt = (
        f"{_SECTION_INSTRUCTION}\n\n"
        f"借款人資料:\n{json.dumps(borrower_data, ensure_ascii=False, default=str, indent=2)}\n\n"
        f"預測結果:\n{json.dumps(prediction, ensure_ascii=False, default=str, indent=2)}\n\n"
        f"SHAP 特徵貢獻（前 {len(shap_payload)} 名）:\n"
        f"{json.dumps(shap_payload, ensure_ascii=False, default=str, indent=2)}"
    )

    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=1500,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            report_text = response.content[0].text
            return report_text.strip() if report_text else fallback
        except Exception as exc:
            print(f"[ERROR] Claude report generation failed on attempt {attempt + 1}: {exc}")
            # Exponential backoff: 1 s, 2 s, 4 s before returning the fallback.
            time.sleep(2**attempt)

    return fallback


# Alias for clarity in import statements that reference the model by name.
generate_credit_report_opus = generate_credit_report
