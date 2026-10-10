import re

GOOD = dict(age="28", sex="Female", height="165", weight="62", activity="Moderately Active", goal="Maintenance")


def token(client, path="/form"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="_csrf" value="([0-9a-f]+)"', html).group(1)


def submit(client, **overrides):
    return client.post("/form", data={**GOOD, "_csrf": token(client), **overrides})


def test_public_pages_render(client):
    for path in ("/", "/form", "/about", "/healthz"):
        assert client.get(path).status_code == 200


def test_protected_pages_redirect_without_profile(client):
    for path in ("/dashboard", "/meals"):
        r = client.get(path)
        assert r.status_code == 302 and r.headers["Location"].endswith("/form")
    assert client.get("/api/plan").status_code == 409


def test_post_redirect_get_flow(client):
    r = submit(client)
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")
    dash = client.get("/dashboard")
    assert dash.status_code == 200 and b"Daily targets" in dash.data
    meals = client.get("/meals")
    assert meals.status_code == 200 and b"Breakfast" in meals.data and b"Snack" in meals.data


def test_invalid_input_shows_errors_and_keeps_values(client):
    r = submit(client, age="12", weight="abc")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Age must be between 18 and 80" in body
    assert "valid number for weight" in body
    assert 'value="165"' in body  # valid fields keep their values


def test_csrf_is_enforced(client):
    r = client.post("/form", data=GOOD)
    assert r.status_code == 400


def test_api_returns_plan_and_meals(client):
    submit(client)
    data = client.get("/api/plan").get_json()
    assert data["plan"]["calories"] > 1200
    assert len(data["meal_plan"]["meals"]) == 4


def test_unknown_route_404_page(client):
    r = client.get("/nope")
    assert r.status_code == 404 and b"could not find" in r.data
