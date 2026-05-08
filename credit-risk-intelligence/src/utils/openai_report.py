"""OpenAI-powered credit risk report generation."""

from __future__ import annotations

import json
import os
import time
from typing import Any


def generate_credit_report(
    borrower_data: dict[str, Any],
    prediction: dict[str, Any],
    shap_top_features: list[tuple[str, float, Any]],
) -> str:
    """Generate a Traditional Chinese credit assessment report.

    Args:
        borrower_data: Borrower feature values used for credit risk evaluation.
        prediction: Prediction output such as risk_score and risk_level.
        shap_top_features: Top SHAP feature tuples as (feature_name, shap_value, raw_value).

    Returns:
        Markdown credit assessment report, or a fallback failure message.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    score = prediction.get("risk_score", "N/A")
    top_3_features = [feature[0] for feature in shap_top_features[:3]]

    payload = {
        "borrower_data": borrower_data,
        "prediction": prediction,
        "shap_top_features": [
            {"feature_name": name, "shap_value": value, "raw_value": raw_value}
            for name, value, raw_value in shap_top_features
        ],
        "instruction": (
            "Output markdown with exactly these sections: ## 借款人摘要, "
            "## 風險評分與等級, ## 主要風險因子（前 3 名 SHAP 特徵的白話解釋）, ## 建議行動."
        ),
    }
    fallback = f"[報告生成失敗] 風險評分: {score} | 主要因子: {top_3_features}"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a senior credit risk analyst at a Taiwanese bank. Generate a concise "
        "credit assessment report in Traditional Chinese (zh-TW)."
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, default=str),
                    },
                ],
            )
            content = response.choices[0].message.content
            return content.strip() if content else fallback
        except Exception as exc:
            print(f"[ERROR] OpenAI report generation failed on attempt {attempt + 1}: {exc}")
            # Exponential backoff: 1s, 2s, then 4s before the final fallback.
            time.sleep(2**attempt)

    return fallback
