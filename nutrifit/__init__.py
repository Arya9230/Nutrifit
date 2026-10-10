"""NutriFit - personalised calorie, macro and meal planning."""
from __future__ import annotations

import hmac
import os
import secrets

from flask import (
    Flask, abort, jsonify, redirect, render_template, request, session, url_for,
)

from .evaluation import load_or_compute
from .nutrition import (
    ACTIVITY_LEVELS, GOALS, Profile, ValidationError, calculate_plan, validate_profile,
)
from .recommender import FoodModel


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        # Set SECRET_KEY in production; the random fallback only keeps dev sessions
        # alive until the process restarts.
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        MAX_CONTENT_LENGTH=16 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    model = FoodModel()  # trained once at start-up, shared read-only
    metrics = load_or_compute(model)
    example_plan = calculate_plan(
        Profile(age=28, sex="Female", height=165.0, weight=62.0,
                activity="Moderately Active", goal="Maintenance")
    )

    # ------------------------------------------------------------ helpers
    def csrf_token() -> str:
        if "_csrf" not in session:
            session["_csrf"] = secrets.token_hex(16)
        return session["_csrf"]

    app.jinja_env.globals["csrf_token"] = csrf_token

    def current_profile() -> Profile | None:
        data = session.get("profile")
        if not data:
            return None
        try:
            return Profile(**data)
        except TypeError:
            session.pop("profile", None)
            return None

    @app.context_processor
    def inject_globals():
        return {"has_profile": "profile" in session, "endpoint": request.endpoint}

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return resp

    # ------------------------------------------------------------- routes
    @app.get("/")
    def home():
        return render_template("home.html", metrics=metrics, example=example_plan)

    @app.route("/form", methods=["GET", "POST"])
    def form():
        errors: dict[str, str] = {}
        values = (current_profile().as_dict() if current_profile() else {}) if request.method == "GET" else request.form
        if request.method == "POST":
            sent = request.form.get("_csrf", "")
            expected = session.get("_csrf", "")
            if not expected or not hmac.compare_digest(sent, expected):
                abort(400, description="Your session expired. Please reload the page and try again.")
            try:
                profile = validate_profile(request.form)
            except ValidationError as exc:
                errors = exc.errors
            else:
                session["profile"] = profile.as_dict()
                session.pop("variant", None)
                return redirect(url_for("dashboard"))  # Post/Redirect/Get
        return render_template(
            "form.html", errors=errors, values=values,
            activities=ACTIVITY_LEVELS, goals=GOALS,
        )

    @app.get("/dashboard")
    def dashboard():
        profile = current_profile()
        if profile is None:
            return redirect(url_for("form"))
        return render_template("dashboard.html", plan=calculate_plan(profile))

    @app.get("/meals")
    def meals():
        profile = current_profile()
        if profile is None:
            return redirect(url_for("form"))
        plan = calculate_plan(profile)
        variant = request.args.get("variant", default=0, type=int) or 0
        variant = max(0, min(variant, 999))
        day = model.plan_day(plan, variant=variant)
        return render_template("meals.html", plan=plan, day=day, variant=variant)

    @app.get("/about")
    def about():
        return render_template("about.html", metrics=metrics)

    @app.get("/api/plan")
    def api_plan():
        """JSON version of the plan, handy for demos and integration tests."""
        profile = current_profile()
        if profile is None:
            return jsonify(error="No profile in session. POST the form at /form first."), 409
        plan = calculate_plan(profile)
        variant = max(0, min(request.args.get("variant", default=0, type=int) or 0, 999))
        return jsonify(plan=plan.as_dict(), meal_plan=model.plan_day(plan, variant=variant))

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok", foods_in_pool=int(len(model.pool)))

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(500)
    def error_page(err):
        code = getattr(err, "code", 500)
        message = {
            400: getattr(err, "description", "Bad request."),
            404: "We could not find that page.",
        }.get(code, "Something went wrong on our side.")
        return render_template("error.html", code=code, message=message), code

    return app
