import numpy as np
import pytest

from nutrifit.nutrition import Profile, calculate_plan
from nutrifit.recommender import SLOTS


def make_plan(goal="Maintenance", sex="Female", weight=62.0):
    return calculate_plan(Profile(28, sex, 165.0, weight, "Moderately Active", goal))


def test_day_plan_is_close_to_targets(model):
    for goal in ("Weight Loss", "Maintenance", "Gain Muscle"):
        plan = make_plan(goal)
        day = model.plan_day(plan)
        assert abs(day["totals"]["Calories"] - plan.calories) / plan.calories < 0.08
        assert abs(day["totals"]["Protein"] - plan.protein_g) / plan.protein_g < 0.30


def test_structure_and_no_repeated_foods(model):
    day = model.plan_day(make_plan())
    assert [m["name"] for m in day["meals"]] == [s[0] for s in SLOTS]
    names = [i["name"] for m in day["meals"] for i in m["items"]]
    assert len(names) == len(set(names))


def test_recommendations_depend_on_the_user(model):
    small = model.plan_day(make_plan(weight=50))
    large = model.plan_day(make_plan("Gain Muscle", "Male", 95))
    assert large["totals"]["Calories"] > small["totals"]["Calories"] * 1.3


def test_deterministic_and_variants_differ(model):
    plan = make_plan()
    a, b = model.plan_day(plan, 0), model.plan_day(plan, 0)
    assert a == b
    other = model.plan_day(plan, 1)
    first = {i["name"] for m in a["meals"] for i in m["items"]}
    second = {i["name"] for m in other["meals"] for i in m["items"]}
    assert first != second


def test_beats_random_baseline(model):
    plan = make_plan()
    rng = np.random.default_rng(0)
    base_errors = [abs(model.random_day(plan, rng)["Calories"] - plan.calories) for _ in range(30)]
    planned = abs(model.plan_day(plan)["totals"]["Calories"] - plan.calories)
    assert planned < np.mean(base_errors) / 3


def test_kmeans_selection_is_valid(model):
    c = model.clusters
    assert 3 <= c.k <= 8
    assert c.silhouette == max(s["silhouette"] for s in c.sweep if s["valid"])
    assert min(c.sizes.values()) >= 5
