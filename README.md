# NutriFit

Personalised daily calorie and macro targets, plus a day of meals sized to hit them.
Built with Flask, pandas and scikit-learn.

## What it does

1. **Targets**: Mifflin-St Jeor BMR x activity factor, adjusted for goal (-20% / maintenance / +10%),
   with a calorie floor, goal-based protein (1.4 / 1.8 / 2.0 g/kg), fat at ~27% and carbs as the remainder.
2. **Food profiling**: K-Means groups foods by per-100 g calories, protein, carbs, fat and fibre.
   *k* is chosen by silhouette score (k = 3..8), rejecting degenerate outlier-only clusters.
3. **Meal planner**: for each of four meals, searches combinations of foods *and portion sizes* and returns
   the one closest to that meal's share of the user's macro targets, with a small bonus for nutrient quality.

## Measured results (200 simulated profiles, seed 7)

| Average gap to daily target | Planner | Random foods |
|---|---|---|
| Calories | 1.3% | 24.5% |
| Protein | 6.1% | 29.6% |
| Carbohydrate | 3.1% | 36.2% |
| Fat | 6.9% | 37.6% |

93.5% of plans land within 5% of the calorie target. K-Means: k = 4, silhouette 0.537.
Reproduce with `python -m nutrifit.evaluation` (writes `artifacts/metrics.json`, which the About page reads).
These numbers measure how well plans match targets, not health outcomes.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000
pytest                   # 45 tests
```

Optional environment variables: `SECRET_KEY` (set this in production), `PORT`, `HOST`, `FLASK_DEBUG=1`.

## Project layout

```
app.py                     entry point
nutrifit/
  __init__.py              app factory, routes, CSRF, error pages, JSON API
  nutrition.py             BMR/TDEE/macros, BMI, input validation
  data.py                  robust CSV cleaning + data-quality report
  catalog.py               curated name fixes for the flattened source table
  recommender.py           K-Means profiling, quality score, macro-matching planner
  evaluation.py            reproducible metrics vs. a random baseline
  templates/ static/       server-rendered UI, hand-written CSS
tests/                     nutrition, data cleaning, recommender, Flask routes
```

Endpoints: `/`, `/form`, `/dashboard`, `/meals?variant=n`, `/about`, `/api/plan` (JSON), `/healthz`.

## Known limits and next steps

- The food table is an old printed handbook (335 rows); swap in USDA FoodData Central for variety and better names.
- No allergy or diet-preference filtering yet.
- The profile lives in a signed session cookie; add accounts and a database to save history.
