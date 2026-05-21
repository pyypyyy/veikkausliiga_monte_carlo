from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import pandas as pd

try:
    from .data_sources import football_data_org
except ImportError:
    from data_sources import football_data_org

EXPECTED_TEAMS = 12
REGULAR_SEASON_ROUNDS_PER_OPPONENT = 2


def canonical_team(name: str) -> str:
    if name in football_data_org.TEAM_ALIASES:
        return football_data_org.TEAM_ALIASES[name]
    raise ValueError(f"Unknown team mapping for API team name: {name}")


def _match_date(match: dict) -> str:
    utc = str(match.get("utcDate", ""))
    return utc[:10] if len(utc) >= 10 else utc


def convert_matches(matches_payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    played_rows: list[dict] = []
    remaining_rows: list[dict] = []

    for match in matches_payload.get("matches", []):
        status = str(match.get("status", "")).upper()
        home = canonical_team(match["homeTeam"]["name"])
        away = canonical_team(match["awayTeam"]["name"])
        date = _match_date(match)

        if status in football_data_org.FINAL_STATUSES:
            full_time = match.get("score", {}).get("fullTime", {})
            hg, ag = full_time.get("home"), full_time.get("away")
            if hg is None or ag is None:
                raise ValueError(f"FINISHED match missing full-time score: {home} vs {away} ({date})")
            played_rows.append(
                {
                    "Date": date,
                    "Home": home,
                    "Away": away,
                    "HomeGoals": int(hg),
                    "AwayGoals": int(ag),
                    "Source": "football-data.org",
                }
            )
        elif status in football_data_org.NON_FINAL_STATUSES:
            remaining_rows.append({"Date": date, "Home": home, "Away": away})

    played_df = pd.DataFrame(played_rows)
    remaining_df = pd.DataFrame(remaining_rows)
    if not played_df.empty:
        played_df = played_df.sort_values(["Date", "Home", "Away"]).reset_index(drop=True)
    if not remaining_df.empty:
        remaining_df = remaining_df.sort_values(["Date", "Home", "Away"]).reset_index(drop=True)
    return played_df, remaining_df


def standings_to_table(standings_payload: dict) -> pd.DataFrame:
    standings = standings_payload.get("standings", [])
    if not standings:
        raise ValueError("No standings found in API response")
    table_rows = standings[0].get("table", [])
    rows = []
    for row in table_rows:
        team = canonical_team(row["team"]["name"])
        rows.append(
            {
                "Team": team,
                "P": int(row["playedGames"]),
                "W": int(row["won"]),
                "D": int(row["draw"]),
                "L": int(row["lost"]),
                "GF": int(row["goalsFor"]),
                "GA": int(row["goalsAgainst"]),
                "GD": int(row["goalDifference"]),
                "Pts": int(row["points"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["Pts", "GD", "GF"], ascending=[False, False, False]).reset_index(drop=True)


def validate_data(current: pd.DataFrame, played: pd.DataFrame, remaining: pd.DataFrame, api_table: pd.DataFrame) -> None:
    canonical_teams = set(current["Team"])

    def check_unique_fixtures(df: pd.DataFrame, name: str) -> None:
        keys = list(zip(df["Home"], df["Away"], df.get("Date", pd.Series([""] * len(df)))))
        dupes = [k for k, c in Counter(keys).items() if c > 1]
        if dupes:
            raise ValueError(f"Duplicate fixtures in {name}: {dupes[:5]}")

    if not played.empty:
        check_unique_fixtures(played, "played_results")
    if not remaining.empty:
        check_unique_fixtures(remaining, "remaining_fixtures")

    overlap = set(zip(played["Home"], played["Away"], played["Date"])) & set(zip(remaining["Home"], remaining["Away"], remaining["Date"]))
    if overlap:
        raise ValueError(f"Fixture appears in both played and remaining sets: {list(overlap)[:5]}")

    for col in ["HomeGoals", "AwayGoals"]:
        if not pd.api.types.is_integer_dtype(played[col]):
            raise ValueError(f"{col} must be integer type")

    played_teams = set(played["Home"]) | set(played["Away"])
    remaining_teams = set(remaining["Home"]) | set(remaining["Away"])
    unknown = (played_teams | remaining_teams) - canonical_teams
    if unknown:
        raise ValueError(f"Non-canonical team names found: {sorted(unknown)}")

    n_teams = len(canonical_teams)
    expected_fixtures = n_teams * (n_teams - 1) // 2 * REGULAR_SEASON_ROUNDS_PER_OPPONENT
    total_fixtures = len(played) + len(remaining)
    if total_fixtures != expected_fixtures:
        raise ValueError(
            f"Fixture count mismatch for regular season: played({len(played)}) + remaining({len(remaining)}) = {total_fixtures}, expected {expected_fixtures}."
        )

    calc_rows = []
    for r in played.to_dict("records"):
        calc_rows.append({"Team": r["Home"], "P": 1, "W": int(r["HomeGoals"] > r["AwayGoals"]), "D": int(r["HomeGoals"] == r["AwayGoals"]), "L": int(r["HomeGoals"] < r["AwayGoals"]), "GF": r["HomeGoals"], "GA": r["AwayGoals"], "Pts": 3 if r["HomeGoals"] > r["AwayGoals"] else 1 if r["HomeGoals"] == r["AwayGoals"] else 0})
        calc_rows.append({"Team": r["Away"], "P": 1, "W": int(r["AwayGoals"] > r["HomeGoals"]), "D": int(r["HomeGoals"] == r["AwayGoals"]), "L": int(r["AwayGoals"] < r["HomeGoals"]), "GF": r["AwayGoals"], "GA": r["HomeGoals"], "Pts": 3 if r["AwayGoals"] > r["HomeGoals"] else 1 if r["HomeGoals"] == r["AwayGoals"] else 0})
    calc = pd.DataFrame(calc_rows).groupby("Team", as_index=False)[["P", "W", "D", "L", "GF", "GA", "Pts"]].sum()
    calc["GD"] = calc["GF"] - calc["GA"]

    merged = api_table.merge(calc, on="Team", suffixes=("_api", "_calc"), how="left").fillna(0)
    mismatches = []
    for metric in ["P", "W", "D", "L", "GF", "GA", "GD", "Pts"]:
        bad = merged[merged[f"{metric}_api"] != merged[f"{metric}_calc"]]
        for _, row in bad.iterrows():
            mismatches.append(f"{row['Team']} {metric}: api={int(row[f'{metric}_api'])}, calc={int(row[f'{metric}_calc'])}")
    if mismatches:
        raise ValueError("Calculated table does not match API standings. Examples: " + "; ".join(mismatches[:8]))


def update_from_football_data_org(season: int) -> None:
    matches = football_data_org.fetch_matches(season)
    standings = football_data_org.fetch_standings(season)
    football_data_org.save_raw(matches, "matches", season)
    football_data_org.save_raw(standings, "standings", season)

    played, remaining = convert_matches(matches)
    table = standings_to_table(standings)
    validate_data(table, played, remaining, table)

    out_dir = Path("data")
    tmp_played = out_dir / "played_results.csv.tmp"
    tmp_remaining = out_dir / "remaining_fixtures.csv.tmp"
    tmp_table = out_dir / "current_table.csv.tmp"

    played.to_csv(tmp_played, index=False)
    remaining.to_csv(tmp_remaining, index=False)
    table.to_csv(tmp_table, index=False)

    tmp_played.replace(out_dir / "played_results.csv")
    tmp_remaining.replace(out_dir / "remaining_fixtures.csv")
    tmp_table.replace(out_dir / "current_table.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Veikkausliiga data from external provider")
    parser.add_argument("--provider", required=True, choices=["football-data-org"])
    parser.add_argument("--season", required=True, type=int)
    args = parser.parse_args()

    if args.provider == "football-data-org":
        update_from_football_data_org(args.season)
        print(f"Updated data files from football-data.org for season {args.season}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Data update failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
