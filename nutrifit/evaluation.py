"""Honest, reproducible evaluation of the models.

Run ``python -m nutrifit.evaluation`` to regenerate ``artifacts/metrics.json``.

What is measured
- K-Means: silhouette score for each candidate k (the chosen k is the best valid one).
- Meal planner: over many random but realistic user profiles, how far the daily
  calories / protein / carbs / fat of the generated plan are from the user's
  targets (mean absolute percentage error), compared with a random baseline that
  picks one food per meal role at its listed serving size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .nutrition import ACTIVITY_LEVELS, GOALS, Profile, calculate_plan
from .recommender import FoodModel

METRICS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "metrics.json"


def random_profile(rng: np.random.Generator) -> Profile:
    return Profile(
        age=int(rng.integers(18, 66)),
        sex=str(rng.choice(["Male", "Female"])),
        height=float(rng.integers(150, 196)),
        weight=float(rng.integers(45, 111)),
        activity=str(rng.choice(list(ACTIVITY_LEVELS))),
        goal=str(rng.choice(list(GOALS))),
    )


def _mape(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - target) / np.maximum(target, 1.0)) * 100)


def evaluate(model: FoodModel, n_profiles: int = 200, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    keys = ("Calories", "Protein", "Carbs", "Fat")
    plan_err = {k: [] for k in keys}
    base_err = {k: [] for k in keys}
    within5 = within10 = 0

    for _ in range(n_profiles):
        plan = calculate_plan(random_profile(rng))
        target = {"Calories": plan.calories, "Protein": plan.protein_g, "Carbs": plan.carbs_g, "Fat": plan.fat_g}
        day = model.plan_day(plan, variant=0)
        base = model.random_day(plan, rng)
        for k in keys:
            plan_err[k].append(abs(day["totals"][k] - target[k]) / max(target[k], 1) * 100)
            base_err[k].append(abs(base[k] - target[k]) / max(target[k], 1) * 100)
        kcal_err = plan_err["Calories"][-1]
        within5 += kcal_err <= 5
        within10 += kcal_err <= 10

    summarise = lambda errs: {k: round(float(np.mean(v)), 1) for k, v in errs.items()}  # noqa: E731
    c = model.clusters
    return {
        "n_profiles": n_profiles,
        "seed": seed,
        "planner_mape_pct": summarise(plan_err),
        "random_baseline_mape_pct": summarise(base_err),
        "calories_within_5pct": round(within5 / n_profiles * 100, 1),
        "calories_within_10pct": round(within10 / n_profiles * 100, 1),
        "kmeans": {
            "chosen_k": c.k,
            "silhouette": c.silhouette,
            "sweep": c.sweep,
            "profiles": [{"name": c.labels[i], "foods": c.sizes[i]} for i in sorted(c.labels)],
        },
        "data_quality": model.report.as_dict(),
    }


def load_or_compute(model: FoodModel) -> dict:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return evaluate(model, n_profiles=40)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    metrics = evaluate(FoodModel(), n_profiles=n)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
