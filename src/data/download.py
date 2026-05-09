"""Download and generate datasets for Credit Risk Intelligence."""

from __future__ import annotations

import argparse
import random
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from faker import Faker


KAGGLE_FIX_INSTRUCTIONS = (
    "Install: pip install kaggle. Setup: place kaggle.json in ~/.kaggle/ "
    "and chmod 600. Get token at https://www.kaggle.com/settings"
)


def download_credit_dataset() -> None:
    """Download and unzip the Give Me Some Credit Kaggle competition dataset.

    Args:
        None.

    Returns:
        None.
    """
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "GiveMeSomeCredit",
        "-p",
        str(raw_dir),
    ]

    try:
        print("[INFO] Downloading GiveMeSomeCredit dataset from Kaggle...")
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] Kaggle CLI failed: {exc}")
        print(f"[ERROR] {KAGGLE_FIX_INSTRUCTIONS}")
        return

    zip_files = sorted(raw_dir.glob("*.zip"))
    if not zip_files:
        print("[ERROR] Kaggle download completed, but no zip file was found in data/raw/.")
        return

    for zip_path in zip_files:
        print(f"[INFO] Unzipping {zip_path} into {raw_dir}...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(raw_dir)

    file_list = sorted(path.name for path in raw_dir.iterdir() if path.is_file())
    print("[INFO] Credit dataset downloaded and extracted successfully.")
    print(f"[INFO] data/raw files: {file_list}")


def generate_synthetic_loan_descriptions(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic borrower loan descriptions and save them as a CSV.

    Args:
        n: Number of synthetic loan descriptions to generate.
        seed: Random seed for reproducible generated text and numeric values.

    Returns:
        A pandas DataFrame with borrower_id, loan_purpose, and description columns.
    """
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "loan_descriptions.csv"

    faker = Faker()
    faker.seed_instance(seed)
    rng = random.Random(seed)

    loan_purposes = ["debt_consolidation", "home_improvement", "medical", "small_business", "other"]
    tones = ["formal", "informal"]

    def money(min_value: int, max_value: int, step: int = 100) -> str:
        value = rng.randrange(min_value, max_value + step, step)
        return f"${value:,}"

    def months() -> int:
        return rng.choice([12, 18, 24, 36, 48, 60, 72])

    templates = {
        "debt_consolidation": {
            "formal": (
                "I am applying for a {amount} loan to consolidate my existing credit card debt. "
                "Currently I have {n_accounts} accounts with total balance of {total}. "
                "My monthly income is {income} and I plan to repay over {months} months."
            ),
            "informal": (
                "I would like a {amount} loan so I can roll my credit card balances into one "
                "payment. I have {n_accounts} accounts totaling about {total}, earn {income} "
                "per month, and expect to pay it back over {months} months."
            ),
        },
        "home_improvement": {
            "formal": (
                "I am requesting a {amount} loan for home improvement work at my residence. "
                "The funds will cover {reason}, and my monthly income is {income}. "
                "I intend to complete repayment within {months} months."
            ),
            "informal": (
                "I need {amount} to take care of {reason} around the house. I bring in {income} "
                "each month and plan to repay the loan over {months} months."
            ),
        },
        "medical": {
            "formal": (
                "I am applying for a {amount} loan to manage medical expenses related to "
                "{reason}. My monthly income is {income}, and I plan to repay the balance over "
                "{months} months."
            ),
            "informal": (
                "I am looking for {amount} to help with {reason}. I make about {income} per "
                "month and can repay it over {months} months."
            ),
        },
        "small_business": {
            "formal": (
                "I am seeking a {amount} loan to support my small business with {reason}. "
                "Current monthly business revenue is {income}, and I plan to repay over "
                "{months} months."
            ),
            "informal": (
                "I need {amount} for my small business, mainly for {reason}. Revenue is around "
                "{income} a month, and I expect to repay the loan in {months} months."
            ),
        },
        "other": {
            "formal": (
                "I am requesting a {amount} personal loan for {reason}. My monthly income is "
                "{income}, and I have budgeted for repayment over {months} months."
            ),
            "informal": (
                "I would like to borrow {amount} for {reason}. I earn about {income} per month "
                "and plan to pay it back over {months} months."
            ),
        },
    }

    reasons = {
        "home_improvement": [
            "roof repairs",
            "kitchen updates",
            "bathroom renovation",
            "new flooring",
            "energy-efficient windows",
        ],
        "medical": [
            "surgery bills",
            "dental treatment",
            "physical therapy",
            "specialist visits",
            "unexpected hospital costs",
        ],
        "small_business": [
            "inventory purchases",
            "equipment upgrades",
            "seasonal working capital",
            "marketing expenses",
            "storefront improvements",
        ],
        "other": [
            "moving expenses",
            "vehicle repairs",
            "education costs",
            "family expenses",
            "an upcoming major purchase",
        ],
    }

    rows: list[dict[str, str]] = []
    for index in range(1, n + 1):
        purpose = rng.choice(loan_purposes)
        tone = rng.choice(tones)

        # Debt consolidation needs account-specific details; the other templates use a reason.
        values = {
            "amount": money(1_000, 40_000, 500),
            "income": money(2_000, 18_000, 100),
            "months": months(),
            "n_accounts": rng.randint(2, 12),
            "total": money(2_000, 75_000, 500),
            "reason": rng.choice(reasons.get(purpose, ["personal expenses"])),
        }
        description = templates[purpose][tone].format(**values)

        # Faker adds small lexical variety without changing the required output schema.
        borrower_id = f"BRW-{faker.unique.random_number(digits=8, fix_len=True)}"
        rows.append(
            {
                "borrower_id": borrower_id,
                "loan_purpose": purpose,
                "description": description,
            }
        )

    dataframe = pd.DataFrame(rows, columns=["borrower_id", "loan_purpose", "description"])
    dataframe.to_csv(output_path, index=False)
    print(f"[INFO] Generated {len(dataframe)} synthetic loan descriptions.")
    print(f"[INFO] Saved CSV to {output_path}.")
    return dataframe


def main() -> None:
    """Parse CLI arguments and run the requested data task.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Download or generate project datasets.")
    parser.add_argument(
        "--target",
        choices=["credit", "text", "all"],
        required=True,
        help="Dataset task to run.",
    )
    args = parser.parse_args()

    if args.target in {"credit", "all"}:
        download_credit_dataset()
    if args.target in {"text", "all"}:
        generate_synthetic_loan_descriptions()


if __name__ == "__main__":
    main()
