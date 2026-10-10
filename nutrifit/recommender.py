"""Food profiling (K-Means) and macro-matching meal planner.

How it works
1. Every food is described by its per-100 g density of calories, protein, carbs,
   fat and fibre. K-Means groups foods into nutrient profiles; k is chosen by
   scanning k = 3..8 and keeping the best silhouette score.
2. A transparent nutrient-quality score (0-100) ranks foods inside the meal pool:
   more protein and fibre per calorie is better, more saturated fat is worse.
3. For every meal slot the planner searches combinations of foods *and* portion
   sizes and returns the one whose calories/protein/carbs/fat are closest to
   the slot's share of the user's daily targets, with a small bonus for quality.
   A "variant" number changes the candidate sample so users can regenerate.

Nothing here is trained on a made-up label: the K-Means result is evaluated with
the silhouette score and the planner is evaluated against a random baseline
(see evaluation.py).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import DataQualityReport, load_food_data
from .nutrition import Plan

CLUSTER_FEATURES = ["Calories_100g", "Protein_100g", "Carbs_100g", "Fat_100g", "Fiber_100g"]
MACRO_COLS = ["Calories", "Protein", "Fat", "Carbs"]

# (slot name, share of daily energy, roles that make up the meal)
SLOTS = [
    ("Breakfast", 0.25, ("dairy", "carb", "fruit")),
    ("Lunch", 0.35, ("protein", "carb", "veg")),
    ("Dinner", 0.30, ("protein", "veg", "carb")),
    ("Snack", 0.10, ("fruit", "nut")),
]

# Portion multipliers the search may apply to the listed serving.
MULTIPLIERS = {
    "protein": (0.5, 0.75, 1.0, 1.5, 2.0),
    "carb": (0.5, 1.0, 1.5, 2.0),
    "veg": (1.0, 2.0, 3.0),
    "fruit": (0.5, 1.0, 2.0),
    "dairy": (0.5, 1.0, 2.0),
    "nut": (0.25, 0.5, 1.0),
}

ROLE_LABEL = {
    "protein": "Protein",
    "carb": "Grain",
    "veg": "Vegetable",
    "fruit": "Fruit",
    "dairy": "Dairy & egg",
    "nut": "Nuts & seeds",
}

CANDIDATES_PER_ROLE = 12


@dataclass
class ClusterInfo:
    k: int
    silhouette: float
    sweep: list[dict]
    labels: dict[int, str]
    sizes: dict[int, int]


class FoodModel:
    def __init__(self, df: pd.DataFrame | None = None, report: DataQualityReport | None = None):
        if df is None:
            df, report = load_food_data()
        self.report = report
        self.df = df.copy()
        self._fit_clusters()
        self._score_quality()
        self.pool = self.df[self.df["InMealPool"]].reset_index(drop=True)
        self.by_role = {r: self.pool.index[self.pool["Role"] == r].to_numpy() for r in MULTIPLIERS}

    # ------------------------------------------------------------------ ML
    def _fit_clusters(self, k_range=range(3, 9)) -> None:
        # Profile the foods the app can actually recommend (not oils or lard,
        # which would form single-item outlier clusters).
        pool_rows = self.df.index[self.df["InMealPool"]]
        scaler = StandardScaler().fit(self.df.loc[pool_rows, CLUSTER_FEATURES])
        X = scaler.transform(self.df.loc[pool_rows, CLUSTER_FEATURES])
        sweep = []
        best = None
        min_size = max(5, int(0.04 * len(X)))  # reject degenerate outlier-only clusters
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            score = float(silhouette_score(X, km.labels_))
            smallest = int(np.bincount(km.labels_).min())
            valid = smallest >= min_size
            sweep.append({"k": k, "silhouette": round(score, 3), "smallest_cluster": smallest, "valid": valid})
            if valid and (best is None or score > best[0]):
                best = (score, k, km)
        score, k, km = best
        self.df["Cluster"] = -1
        self.df.loc[pool_rows, "Cluster"] = km.labels_
        centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=CLUSTER_FEATURES)
        labels = {i: _label_centroid(row) for i, row in centroids.iterrows()}
        labels = _dedupe_labels(labels, centroids)
        self.df["Profile"] = self.df["Cluster"].map(labels).fillna("Not in meal pool")
        self.clusters = ClusterInfo(
            k=k,
            silhouette=round(score, 3),
            sweep=sweep,
            labels=labels,
            sizes={int(c): int(n) for c, n in pd.Series(km.labels_).value_counts().sort_index().items()},
        )

    def _score_quality(self) -> None:
        """Transparent 0-100 nutrient-quality score (percentile of a density formula)."""
        per_100kcal = lambda col: self.df[col] / self.df["Calories"] * 100  # noqa: E731
        raw = 2.0 * per_100kcal("Protein") + 3.0 * per_100kcal("Fiber") - 1.5 * per_100kcal("Sat.Fat")
        self.df["Quality"] = (raw.rank(pct=True) * 100).round().astype(int)

    # ------------------------------------------------------------ planning
    def plan_day(self, plan: Plan, variant: int = 0) -> dict:
        used: set[int] = set()
        meals = []
        for slot_index, (name, share, roles) in enumerate(SLOTS):
            target = {
                "Calories": plan.calories * share,
                "Protein": plan.protein_g * share,
                "Fat": plan.fat_g * share,
                "Carbs": plan.carbs_g * share,
            }
            rng = np.random.default_rng(10_000 * (variant + 1) + slot_index)
            chosen = self._best_combo(roles, target, rng, used)
            used.update(c["index"] for c in chosen)
            meals.append(self._format_meal(name, chosen, target))

        totals = {k: sum(m["totals"][k] for m in meals) for k in ("Calories", "Protein", "Fat", "Carbs")}
        daily_target = {
            "Calories": plan.calories,
            "Protein": plan.protein_g,
            "Fat": plan.fat_g,
            "Carbs": plan.carbs_g,
        }
        return {"meals": meals, "totals": totals, "target": daily_target, "variant": variant}

    def _candidates(self, role: str, rng: np.random.Generator, used: set[int]) -> np.ndarray:
        idx = self.by_role[role]
        idx = np.array([i for i in idx if i not in used])
        quality = self.pool.loc[idx, "Quality"].to_numpy()
        # Prefer the better half by quality, then sample for variety.
        keep = idx[quality >= np.median(quality)]
        if len(keep) < CANDIDATES_PER_ROLE:
            keep = idx
        size = min(CANDIDATES_PER_ROLE, len(keep))
        return rng.choice(keep, size=size, replace=False)

    def _best_combo(self, roles, target, rng, used) -> list[dict]:
        option_sets = []  # per role: (macros [n,4], meta list)
        for role in roles:
            cands = self._candidates(role, rng, used)
            macros, meta = [], []
            for i in cands:
                row = self.pool.loc[i]
                for m in MULTIPLIERS[role]:
                    macros.append([row[c] * m for c in MACRO_COLS])
                    meta.append((int(i), role, m))
            option_sets.append((np.array(macros), meta))

        t = np.array([target[c] for c in MACRO_COLS])
        sizes = [len(m) for _, m in option_sets]
        # Sum macros over every combination via broadcasting.
        total = np.zeros(sizes + [4])
        for axis, (macros, _) in enumerate(option_sets):
            shape = [1] * len(sizes) + [4]
            shape[axis] = sizes[axis]
            total = total + macros.reshape(shape)

        rel = np.abs(total - t) / np.maximum(t, 1.0)
        # weights: calories matter most, then protein, then carbs/fat.
        weights = np.array([2.0, 1.0, 0.5, 0.5])
        cost = (rel * weights).sum(axis=-1)

        # Mild preferences: nutrient quality up, extreme portions down.
        for axis, (_, meta) in enumerate(option_sets):
            q = np.array([self.pool.at[i, "Quality"] for i, _, _ in meta]) / 100.0
            pen = np.array([abs(np.log(m)) for _, _, m in meta])
            shape = [1] * len(sizes)
            shape[axis] = sizes[axis]
            cost = cost - 0.05 * q.reshape(shape) + 0.04 * pen.reshape(shape)

        flat = int(np.argmin(cost))
        picks = np.unravel_index(flat, sizes)
        return [
            {"index": option_sets[a][1][p][0], "role": option_sets[a][1][p][1], "mult": option_sets[a][1][p][2]}
            for a, p in enumerate(picks)
        ]

    def _format_meal(self, name: str, chosen: list[dict], target: dict) -> dict:
        items = []
        totals = dict.fromkeys(MACRO_COLS, 0.0)
        for c in chosen:
            row = self.pool.loc[c["index"]]
            m = c["mult"]
            item = {
                "name": row["Display"],
                "role": ROLE_LABEL[c["role"]],
                "portion": row["Serving"] if m == 1 else f"{m:g} × {row['Serving']}",
                "grams": int(round(row["Grams"] * m)),
                "Calories": int(round(row["Calories"] * m)),
                "Protein": float(round(row["Protein"] * m, 1)),
                "Fat": float(round(row["Fat"] * m, 1)),
                "Carbs": float(round(row["Carbs"] * m, 1)),
                "profile": row["Profile"],
                "quality": int(row["Quality"]),
            }
            for col in MACRO_COLS:
                totals[col] += row[col] * m
            items.append(item)
        totals = {k: float(round(v, 1)) for k, v in totals.items()}
        totals["Calories"] = int(round(totals["Calories"]))
        return {
            "name": name,
            "items": items,
            "totals": totals,
            "target": {k: float(round(v, 1)) for k, v in target.items()},
            "kcal_error_pct": round((totals["Calories"] - target["Calories"]) / target["Calories"] * 100, 1),
            "quality": int(round(np.mean([i["quality"] for i in items]))),
        }

    # -------------------------------------------------------- random baseline
    def random_day(self, plan: Plan, rng: np.random.Generator) -> dict:
        """Baseline: one random food per role at its listed serving, no optimisation."""
        totals = dict.fromkeys(MACRO_COLS, 0.0)
        for _, _, roles in SLOTS:
            for role in roles:
                row = self.pool.loc[rng.choice(self.by_role[role])]
                for col in MACRO_COLS:
                    totals[col] += row[col]
        return totals


# ---------------------------------------------------------------------------
def _label_centroid(c: pd.Series) -> str:
    kcal, p, carb, fat = c["Calories_100g"], c["Protein_100g"], c["Carbs_100g"], c["Fat_100g"]
    energy = max(4 * p + 4 * carb + 9 * fat, 1e-9)
    p_share, c_share, f_share = 4 * p / energy, 4 * carb / energy, 9 * fat / energy
    if kcal < 90:
        return "Low calorie density"
    if f_share > 0.65:
        return "Fat-dense"
    if f_share > 0.40 and p_share >= 0.15:
        return "Protein & fat"
    if p_share > 0.30:
        return "High protein"
    if c_share > 0.62:
        return "Carb-rich"
    return "Balanced"


def _dedupe_labels(labels: dict[int, str], centroids: pd.DataFrame) -> dict[int, str]:
    """If two clusters earn the same name, tell them apart by energy density."""
    out = dict(labels)
    for name in set(labels.values()):
        members = [i for i, n in labels.items() if n == name]
        if len(members) > 1:
            members.sort(key=lambda i: centroids.loc[i, "Calories_100g"])
            for rank, i in enumerate(members):
                tag = ("lighter", "mid", "denser")[min(rank, 2)] if len(members) == 3 else ("lighter", "denser")[min(rank, 1)]
                out[i] = f"{name} ({tag})"
    return out
