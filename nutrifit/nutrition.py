"""Energy and macronutrient targets from a user profile.

References
- BMR: Mifflin-St Jeor equation (Mifflin et al., 1990).
- Activity multipliers: standard Harris-Benedict PAL factors.
- BMI categories: WHO adult classification.
- Protein per kg of body weight follows the range commonly cited in sports
  nutrition position statements (about 1.2-2.0 g/kg depending on goal).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

ACTIVITY_LEVELS = {
    "Sedentary": (1.2, "Little or no exercise"),
    "Lightly Active": (1.375, "Light exercise 1-3 days/week"),
    "Moderately Active": (1.55, "Moderate exercise 3-5 days/week"),
    "Very Active": (1.725, "Hard exercise 6-7 days/week"),
    "Extra Active": (1.9, "Very heavy training or a physical job"),
}

GOALS = {
    "Weight Loss": "Lose fat at a sustainable pace",
    "Maintenance": "Hold your current weight",
    "Gain Muscle": "Build muscle with a small surplus",
}

SEXES = ("Male", "Female")

# Daily floor so a deficit never drops into an unsafe range.
MIN_CALORIES = {"Male": 1500, "Female": 1200}

LIMITS = {
    "age": (18, 80),
    "height": (120.0, 230.0),
    "weight": (30.0, 250.0),
}


class ValidationError(ValueError):
    """Raised with a dict of field -> message."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


@dataclass(frozen=True)
class Profile:
    age: int
    sex: str
    height: float  # cm
    weight: float  # kg
    activity: str
    goal: str

    def as_dict(self) -> dict:
        return asdict(self)


def validate_profile(form: dict) -> Profile:
    """Turn raw form values into a Profile or raise ValidationError."""
    errors: dict[str, str] = {}

    def number(name: str, label: str, cast):
        raw = str(form.get(name, "")).strip()
        lo, hi = LIMITS[name]
        try:
            value = cast(raw)
        except ValueError:
            errors[name] = f"Enter a valid number for {label.lower()}."
            return None
        if not lo <= value <= hi:
            unit = {"age": "years", "height": "cm", "weight": "kg"}[name]
            errors[name] = f"{label} must be between {lo:g} and {hi:g} {unit}."
            return None
        return value

    age = number("age", "Age", int)
    height = number("height", "Height", float)
    weight = number("weight", "Weight", float)

    sex = str(form.get("sex", "")).strip().title()
    if sex not in SEXES:
        errors["sex"] = "Choose male or female (used by the BMR formula)."
    activity = str(form.get("activity", "")).strip()
    if activity not in ACTIVITY_LEVELS:
        errors["activity"] = "Choose an activity level."
    goal = str(form.get("goal", "")).strip()
    if goal not in GOALS:
        errors["goal"] = "Choose a goal."

    if errors:
        raise ValidationError(errors)
    return Profile(age, sex, height, weight, activity, goal)  # type: ignore[arg-type]


def bmi_category(bmi: float) -> tuple[str, str]:
    """Return (label, tone) where tone drives the colour in the UI."""
    if bmi < 18.5:
        return "Underweight", "warn"
    if bmi < 25:
        return "Healthy weight", "good"
    if bmi < 30:
        return "Overweight", "warn"
    return "Obese", "bad"


@dataclass
class Plan:
    profile: Profile
    bmi: float
    bmi_label: str
    bmi_tone: str
    bmr: int
    tdee: int
    calories: int
    adjustment: int  # kcal/day vs maintenance (negative = deficit)
    protein_g: int
    fat_g: int
    carbs_g: int
    fiber_g: int
    water_l: float
    weekly_change_kg: float
    healthy_range_kg: tuple[float, float]
    macro_pct: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["profile"] = self.profile.as_dict()
        return d


def calculate_plan(p: Profile) -> Plan:
    height_m = p.height / 100
    bmi = round(p.weight / height_m**2, 1)
    bmi_label, bmi_tone = bmi_category(bmi)

    # Mifflin-St Jeor
    bmr = 10 * p.weight + 6.25 * p.height - 5 * p.age + (5 if p.sex == "Male" else -161)
    tdee = bmr * ACTIVITY_LEVELS[p.activity][0]

    if p.goal == "Weight Loss":
        target = tdee * 0.80  # ~20 % deficit
        protein_per_kg = 1.8
    elif p.goal == "Gain Muscle":
        target = tdee * 1.10  # ~10 % surplus
        protein_per_kg = 2.0
    else:
        target = tdee
        protein_per_kg = 1.4
    target = max(target, MIN_CALORIES[p.sex])
    calories = int(round(target / 10.0) * 10)

    # Protein by body weight, capped at 35 % of energy so it stays realistic
    # for small or lightly-active people on a deficit.
    protein_g = min(p.weight * protein_per_kg, 0.35 * calories / 4)
    fat_g = 0.27 * calories / 9
    carbs_g = max(0.0, (calories - protein_g * 4 - fat_g * 9) / 4)

    protein_g, fat_g, carbs_g = (int(round(x)) for x in (protein_g, fat_g, carbs_g))
    energy = protein_g * 4 + fat_g * 9 + carbs_g * 4 or 1
    macro_pct = {
        "protein": round(protein_g * 4 / energy * 100),
        "fat": round(fat_g * 9 / energy * 100),
        "carbs": round(carbs_g * 4 / energy * 100),
    }

    adjustment = calories - int(round(tdee))
    # ~7700 kcal per kg of body fat.
    weekly_change = round(adjustment * 7 / 7700, 2)

    return Plan(
        profile=p,
        bmi=bmi,
        bmi_label=bmi_label,
        bmi_tone=bmi_tone,
        bmr=int(round(bmr)),
        tdee=int(round(tdee)),
        calories=calories,
        adjustment=adjustment,
        protein_g=protein_g,
        fat_g=fat_g,
        carbs_g=carbs_g,
        fiber_g=int(round(14 * calories / 1000)),
        water_l=round(p.weight * 0.035, 1),
        weekly_change_kg=weekly_change,
        healthy_range_kg=(round(18.5 * height_m**2, 1), round(24.9 * height_m**2, 1)),
        macro_pct=macro_pct,
    )
