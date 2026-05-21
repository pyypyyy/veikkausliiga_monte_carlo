from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
from urllib import parse, request

BASE_URL = "https://api.football-data.org/v4"
COMPETITION_CODE = "VEI"
COMPETITION_ID = 2031

# Map API names to canonical simulator names.
TEAM_ALIASES: dict[str, str] = {
    "AC Oulu": "AC Oulu",
    "FC Inter Turku": "FC Inter",
    "FC Haka": "Haka",
    "FF Jaro": "Jaro",
    "Gnistan Helsinki": "Gnistan",
    "HJK Helsinki": "HJK",
    "IFK Mariehamn": "IFK Mariehamn",
    "Ilves Tampere": "Ilves",
    "KTP Kotka": "KTP",
    "KuPS Kuopio": "KuPS",
    "SJK Seinäjoki": "SJK",
    "Vaasan Palloseura": "VPS",
}

NON_FINAL_STATUSES = {"SCHEDULED", "TIMED", "POSTPONED", "IN_PLAY", "PAUSED", "SUSPENDED"}
FINAL_STATUSES = {"FINISHED"}


def _api_get(path: str, params: dict[str, object] | None = None) -> dict:
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("FOOTBALL_DATA_API_KEY is not set. Please export your football-data.org API token.")

    query = f"?{parse.urlencode(params)}" if params else ""
    req = request.Request(
        f"{BASE_URL}{path}{query}",
        headers={"X-Auth-Token": api_key, "Accept": "application/json"},
        method="GET",
    )
    with request.urlopen(req, timeout=30) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))
    return payload


def fetch_matches(season: int) -> dict:
    return _api_get(f"/competitions/{COMPETITION_CODE}/matches", {"season": season})


def fetch_standings(season: int) -> dict:
    return _api_get(f"/competitions/{COMPETITION_CODE}/standings", {"season": season})


def fetch_teams(season: int) -> dict:
    return _api_get(f"/competitions/{COMPETITION_CODE}/teams", {"season": season})


def save_raw(payload: dict, kind: str, season: int, output_dir: str | Path = "data/raw") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"football_data_org_{kind}_{season}_{ts}.json"
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path
