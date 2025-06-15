from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.cache import cache
from .utils import retrieve_auth, verify_auth, compress, decompress
from .plots import make_plots
from os import getenv
import plotly.io as pio
from cryptography.fernet import Fernet

fernet = Fernet(getenv("FERNET_KEY"))
PASSWORD_COOKIE_DURATION = 60 * 60 * 24 * 7 * 31  # 1 month
SESSION_COOKIE_AGE = 60 * 60 * 2  # 2 hours


def landing_page(request):
    return render(request, "landing.html")


def login_page(request):
    if request.session.get("username", "") and request.COOKIES.get("remember_me"):
        return redirect("dashboard")

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        remember_password = (
            request.POST.get("remember_me") == "on"
        )  # checkbox for "Remember me"

        if verify_auth(username, password):
            request.session["username"] = username
            request.session.set_expiry(PASSWORD_COOKIE_DURATION)

            response = redirect("dashboard")

            if remember_password:
                # Encrypt the password and store cookie
                encrypted_password = fernet.encrypt(password.encode()).decode()
                response.set_cookie(
                    "remember_me",
                    encrypted_password,
                    max_age=PASSWORD_COOKIE_DURATION,
                    secure=True,
                    httponly=True,
                    samesite="Lax",
                )
            else:
                # Session cookie (no max_age) — cookie lasts until browser closes
                encrypted_password = fernet.encrypt(password.encode()).decode()
                response.set_cookie(
                    "remember_me",
                    encrypted_password,
                    secure=True,
                    httponly=True,
                    samesite="Lax",
                )

            return response
        else:
            messages.error(request, "Invalid login")

    else:
        # Prefill form if cookies exist
        username = request.session.get("username", "")

    return render(request, "login_form.html", {"username": username})


def dashboard(request):
    from django.core.cache import cache

    # Check login session or cookie
    if "username" not in request.session and not request.COOKIES.get("remember_me"):
        return redirect("login")

    # Handle logout
    if request.method == "POST" and "logout" in request.POST:
        response = redirect("login")
        session_key = request.session.session_key
        request.session.flush()
        response.delete_cookie("remember_me")

        if session_key:
            cache.delete(f"user_plots_{session_key}")
        return response

    session_key = request.session.session_key
    cache_key = f"user_plots_{session_key}" if session_key else None

    plots = []

    # If user posts a plot generation request
    if request.method == "POST" and "logout" not in request.POST:
        start_date = request.POST.get("start_date")
        end_date = request.POST.get("end_date")
        username = request.session.get("username")
        encrypted_password = request.COOKIES.get("remember_me")

        if not (username and encrypted_password):
            return redirect("login")

        password = fernet.decrypt(encrypted_password.encode()).decode()
        auth = retrieve_auth(username, password)
        assert auth

        figures = make_plots(auth, start_date, end_date)
        plots = [
            pio.to_html(fig, full_html=False, include_plotlyjs=False) for fig in figures
        ]

        if cache_key:
            cache.set(cache_key, compress(plots), timeout=SESSION_COOKIE_AGE)  # 2 hours

    # On GET, try to retrieve cached plots
    elif cache_key:
        cached = cache.get(cache_key)
        if cached:
            plots = decompress(cached)

    return render(request, "dashboard.html", {"plots": plots})


def example_page(request):
    cached = cache.get("weekly_plots")
    if cached:
        plots = decompress(cached)
    else:
        username = getenv("DEV_USERNAME")
        password = getenv("DEV_PASSWORD")
        assert username and password

        auth = retrieve_auth(username, password)
        figures = make_plots(auth, None, None)

        plots = [
            pio.to_html(fig, full_html=False, include_plotlyjs=False) for fig in figures
        ]

        cache.set("weekly_plots", compress(plots), timeout=60 * 60 * 24 * 7)

    return render(request, "example.html", {"plots": plots})
