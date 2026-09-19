"""Rule-based seeds and bounded cross-family exploration candidates."""

from __future__ import annotations

from src.factors.schema import FactorSpec


def seed_candidates() -> list[FactorSpec]:
    candidates: list[FactorSpec] = []
    for window in (1, 5, 20, 60):
        for operator in ("rank", "zscore"):
            candidates.append(
                FactorSpec(
                    factor_id=f"price_momentum_{window}d_{operator}",
                    generation=0,
                    parent_ids=(),
                    family="price",
                    hypothesis=(
                        "Recent relative winners may continue to outperform because information "
                        "diffuses gradually across investors."
                    ),
                    base_feature="return",
                    window=window,
                    cs_operator=operator,
                )
            )
    for window in (5, 20, 60):
        candidates.append(
            FactorSpec(
                factor_id=f"price_momentum_{window}d_vol_adjusted",
                generation=0,
                parent_ids=(),
                family="price",
                hypothesis="Momentum scaled by recent risk may isolate persistent price pressure.",
                base_feature="return",
                ts_operator="vol_adjust",
                window=window,
                cs_operator="rank",
            )
        )
    for window in (5, 20, 60):
        for operator in ("rank", "zscore"):
            candidates.append(
                FactorSpec(
                    factor_id=f"price_low_vol_{window}d_{operator}",
                    generation=0,
                    parent_ids=(),
                    family="price",
                    hypothesis="Lower idiosyncratic volatility may proxy for safer persistent demand.",
                    base_feature="volatility",
                    window=window,
                    cs_operator=operator,
                    direction=-1,
                )
            )
    for window in (5, 20):
        for operator in ("rank", "zscore"):
            candidates.append(
                FactorSpec(
                    factor_id=f"price_volume_shock_{window}d_{operator}",
                    generation=0,
                    parent_ids=(),
                    family="price",
                    hypothesis="Unusual trading intensity may reveal attention and order-flow shocks.",
                    base_feature="volume_shock",
                    window=window,
                    cs_operator=operator,
                )
            )
    for window in (20, 60):
        candidates.append(
            FactorSpec(
                factor_id=f"price_distance_to_high_{window}d",
                generation=0,
                parent_ids=(),
                family="price",
                hypothesis="Distance from a recent high may capture trend persistence or underreaction.",
                base_feature="distance_to_high",
                window=window,
                cs_operator="rank",
            )
        )
    fundamental_hypotheses = {
        "after_tax_roe": "Profitable firms may earn higher returns if profitability is underpriced.",
        "operating_margin": "Strong operating efficiency may predict resilient future cash flows.",
        "profit_margin": "High margins may identify firms with durable pricing power.",
        "earnings_yield": "Cheap firms relative to current earnings may offer a valuation premium.",
        "asset_growth": "Aggressive asset growth may precede lower returns through overinvestment.",
    }
    for feature, hypothesis in fundamental_hypotheses.items():
        candidates.append(
            FactorSpec(
                factor_id=f"fundamental_{feature}",
                generation=0,
                parent_ids=(),
                family="fundamental",
                hypothesis=hypothesis,
                base_feature=feature,
                cs_operator="winsorize_zscore",
                direction=-1 if feature == "asset_growth" else 1,
            )
        )
    for window in (5, 20):
        candidates.append(
            FactorSpec(
                factor_id=f"news_volume_{window}d",
                generation=0,
                parent_ids=(),
                family="news",
                hypothesis="Sustained news attention may proxy for gradual information incorporation.",
                base_feature="news_volume",
                window=window,
                cs_operator="zscore",
            )
        )
    candidates.append(
        FactorSpec(
            factor_id="DEMO_leaky_future_return_5d",
            generation=0,
            parent_ids=(),
            family="integrity_demo",
            hypothesis="Demonstration only: future returns will look predictive because they are the target.",
            base_feature="future_return",
            window=5,
            cs_operator="rank",
            demonstration_only=True,
        )
    )
    return candidates


def exploration_candidates(generation: int) -> list[FactorSpec]:
    return [
        FactorSpec(
            factor_id=f"g{generation}_momentum20_x_news5",
            generation=generation,
            parent_ids=(),
            family="interaction",
            hypothesis=(
                "News attention may condition the strength of medium-horizon price continuation."
            ),
            base_feature="return",
            window=20,
            cs_operator="zscore",
            interaction_feature="news_volume",
            interaction_window=5,
            mutation_reason="cross-family exploration",
        ),
        FactorSpec(
            factor_id=f"g{generation}_earnings_yield_x_news20",
            generation=generation,
            parent_ids=(),
            family="interaction",
            hypothesis="News attention may accelerate correction of valuation mispricing.",
            base_feature="earnings_yield",
            cs_operator="zscore",
            interaction_feature="news_volume",
            interaction_window=20,
            mutation_reason="under-tested fundamental-news interaction",
        ),
    ]
