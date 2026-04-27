"""
keep_alive.py — Prevents Streamlit app from sleeping on free hosting.
Runs a background thread that pings the app URL every 5 minutes.
"""

import threading
import time
import requests
import os


def ping_self(url: str, interval_seconds: int = 300):
    """
    Continuously pings the given URL at a fixed interval to keep the app awake.
    :param url: The full URL of the Streamlit app (e.g. https://yourapp.streamlit.app)
    :param interval_seconds: How often to ping (default: 300s = 5 minutes)
    """
    while True:
        try:
            response = requests.get(url, timeout=10)
            print(f"[KeepAlive] Pinged {url} → Status: {response.status_code}")
        except Exception as e:
            print(f"[KeepAlive] Ping failed: {e}")
        time.sleep(interval_seconds)


def start_keep_alive():
    """
    Starts the keep-alive background thread.
    Reads APP_URL from environment variables (set in .env).
    If APP_URL is not set, keep-alive is silently skipped (safe for local dev).
    """
    app_url = os.getenv("APP_URL", "").strip()

    if not app_url:
        print("[KeepAlive] APP_URL not set — skipping keep-alive (OK for local use).")
        return

    thread = threading.Thread(
        target=ping_self,
        args=(app_url, 300),   # Ping every 5 minutes
        daemon=True            # Dies automatically when app stops
    )
    thread.start()
    print(f"[KeepAlive] 🟢 Keep-alive thread started → pinging {app_url} every 5 minutes.")
