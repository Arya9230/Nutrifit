import pytest

from nutrifit.nutrition import Profile, ValidationError, calculate_plan, validate_profile

GOOD = dict(age="30", sex="Male", height="180", weight="80", activity="Moderately Active", goal="Maintenance")


def test_mifflin_st_jeor_known_value():
    # 10*80 + 6.25*180 - 5*30 + 5 = 1780 ; x1.55 = 2759
    plan = calculate_plan(validate_profile(GOOD))
    assert plan.bmr == 1780
    assert plan.tdee == 2759
    assert plan.calories == 2760  # rounded to nearest 10


def test_female_constant_is_lower():
    m = calculate_plan(validate_profile(GOOD))
    f = calculate_plan(validate_profile({**GOOD, "sex": "Female"}))
    assert m.bmr - f.bmr == 166


def test_goal_adjusts_calories():
    base = calculate_plan(validate_profile(GOOD)).calories
    loss = calculate_plan(validate_profile({**GOOD, "goal": "Weight Loss"}))
    gain = calculate_plan(validate_profile({**GOOD, "goal": "Gain Muscle"}))
    assert loss.calories < base < gain.calories
    assert loss.adjustment < 0 < gain.adjustment
    assert loss.weekly_change_kg < 0 < gain.weekly_change_kg


def test_calorie_floor_protects_small_sedentary_user():
    plan = calculate_plan(validate_profile(
        {**GOOD, "sex": "Female", "age": "75", "height": "145", "weight": "40",
         "activity": "Sedentary", "goal": "Weight Loss"}))
    assert plan.calories >= 1200


def test_macros_add_up_to_calories():
    plan = calculate_plan(validate_profile(GOOD))
    energy = plan.protein_g * 4 + plan.carbs_g * 4 + plan.fat_g * 9
    assert abs(energy - plan.calories) / plan.calories < 0.03
    assert sum(plan.macro_pct.values()) in (99, 100, 101)


@pytest.mark.parametrize("bmi_weight,label", [(50, "Underweight"), (70, "Healthy weight"), (85, "Overweight"), (110, "Obese")])
def test_bmi_categories(bmi_weight, label):
    plan = calculate_plan(validate_profile({**GOOD, "weight": str(bmi_weight)}))
    assert plan.bmi_label == label


@pytest.mark.parametrize("field,value", [
    ("age", "17"), ("age", "abc"), ("height", "0"), ("height", "300"),
    ("weight", "-5"), ("weight", ""), ("sex", "Robot"), ("activity", "Couch"), ("goal", "Fly"),
])
def test_validation_rejects_bad_input(field, value):
    with pytest.raises(ValidationError) as exc:
        validate_profile({**GOOD, field: value})
    assert field in exc.value.errors


def test_validation_collects_all_errors():
    with pytest.raises(ValidationError) as exc:
        validate_profile({})
    assert {"age", "height", "weight", "sex", "activity", "goal"} <= set(exc.value.errors)
