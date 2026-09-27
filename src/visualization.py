"""Reusable charts for exploratory analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def subscription_rate_by_category(
    df: pd.DataFrame, category: str, target: str = "target_subscription"
) -> pd.DataFrame:
    """Calculate counts and subscription rate for every category value."""
    summary = (
        df.groupby(category, dropna=False)[target]
        .agg(customer_count="size", subscription_rate="mean")
        .sort_values("subscription_rate", ascending=False)
        .reset_index()
    )
    return summary


def save_subscription_rate_chart(
    summary: pd.DataFrame, category: str, output_path: str | Path
) -> None:
    """Save a sorted category-level subscription-rate bar chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(summary) * 0.42)))
    sns.barplot(data=summary, y=category, x="subscription_rate", ax=ax, color="#4C78A8")
    ax.set(
        title=f"Subscription Rate by {category.replace('_', ' ').title()}",
        xlabel="Subscription Rate",
        ylabel=category.replace("_", " ").title(),
        xlim=(0, max(0.05, summary["subscription_rate"].max() * 1.18)),
    )
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.1%}" for v in summary["subscription_rate"]], padding=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
