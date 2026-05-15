from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationResult:
    summary: pd.DataFrame
    position_distribution: pd.DataFrame
    target_points: pd.DataFrame
    target_points_by_team: pd.DataFrame
    team_parameters: pd.DataFrame
    fixture_model: pd.DataFrame


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_inputs(config: dict[str, Any], base_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(base_dir)
    table_path = base / config["data"]["current_table_csv"]
    fixture_path = base / config["data"]["remaining_fixtures_csv"]

    current = pd.read_csv(table_path)
    fixtures = pd.read_csv(fixture_path)

    required_table_cols = {"Team", "P", "W", "D", "L", "GF", "GA", "Pts"}
    required_fixture_cols = {"Home", "Away"}

    missing_table = required_table_cols - set(current.columns)
    missing_fixture = required_fixture_cols - set(fixtures.columns)

    if missing_table:
        raise ValueError(f"current_table.csv is missing columns: {sorted(missing_table)}")
    if missing_fixture:
        raise ValueError(f"remaining_fixtures.csv is missing columns: {sorted(missing_fixture)}")

    for col in ["P", "W", "D", "L", "GF", "GA", "Pts"]:
        current[col] = pd.to_numeric(current[col], errors="raise")

    if "GD" not in current.columns:
        current["GD"] = current["GF"] - current["GA"]
    else:
        current["GD"] = pd.to_numeric(current["GD"], errors="raise")

    teams = set(current["Team"])
    fixture_teams = set(fixtures["Home"]) | set(fixtures["Away"])
    unknown = fixture_teams - teams
    if unknown:
        raise ValueError(f"Fixtures contain teams not found in current_table.csv: {sorted(unknown)}")

    return current, fixtures


def compute_team_parameters(current: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    model = config["model"]
    prior_goals = float(model["prior_matches_goals"])
    prior_form = float(model["prior_matches_form"])

    total_team_games = float(current["P"].sum())
    if total_team_games <= 0:
        raise ValueError("Total played matches must be greater than zero.")

    league_gf_pg = float(current["GF"].sum() / total_team_games)
    league_ga_pg = float(current["GA"].sum() / total_team_games)
    league_win_rate = float(current["W"].sum() / total_team_games)
    league_draw_rate = float(current["D"].sum() / total_team_games)
    league_loss_rate = float(current["L"].sum() / total_team_games)

    rows = []
    for row in current.to_dict("records"):
        played = float(row["P"])
        team = row["Team"]

        gf_pg = (row["GF"] + prior_goals * league_gf_pg) / (played + prior_goals)
        ga_pg = (row["GA"] + prior_goals * league_ga_pg) / (played + prior_goals)

        attack_strength = gf_pg / league_gf_pg if league_gf_pg > 0 else 1.0
        defence_weakness = ga_pg / league_ga_pg if league_ga_pg > 0 else 1.0

        adj_win = (row["W"] + prior_form * league_win_rate) / (played + prior_form)
        adj_draw = (row["D"] + prior_form * league_draw_rate) / (played + prior_form)
        adj_loss = (row["L"] + prior_form * league_loss_rate) / (played + prior_form)

        win_index = adj_win / league_win_rate if league_win_rate > 0 else 1.0
        draw_index = adj_draw / league_draw_rate if league_draw_rate > 0 else 1.0
        loss_resistance = (1.0 - adj_loss) / (1.0 - league_loss_rate) if league_loss_rate < 1 else 1.0
        volatility = (adj_win + adj_loss) / (league_win_rate + league_loss_rate) if (league_win_rate + league_loss_rate) > 0 else 1.0

        rows.append(
            {
                "Team": team,
                "played": played,
                "attack_strength": attack_strength,
                "defence_weakness": defence_weakness,
                "adjusted_win_rate": adj_win,
                "adjusted_draw_rate": adj_draw,
                "adjusted_loss_rate": adj_loss,
                "win_index": win_index,
                "draw_index": draw_index,
                "loss_resistance": loss_resistance,
                "volatility": volatility,
            }
        )

    return pd.DataFrame(rows)


def poisson_pmf(lmbda: float, max_goals: int) -> np.ndarray:
    goals = np.arange(max_goals + 1, dtype=np.int64)
    pmf = np.empty(max_goals + 1, dtype=np.float64)
    pmf[0] = np.exp(-lmbda)
    for g in range(1, max_goals + 1):
        pmf[g] = pmf[g - 1] * lmbda / g
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


def build_score_distribution(
    home_lambda: float,
    away_lambda: float,
    home_params: dict[str, float],
    away_params: dict[str, float],
    model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    max_goals = int(model["max_goals_per_team"])
    home_pmf = poisson_pmf(home_lambda, max_goals)
    away_pmf = poisson_pmf(away_lambda, max_goals)
    matrix = np.outer(home_pmf, away_pmf)

    home_goals_grid, away_goals_grid = np.meshgrid(
        np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij"
    )

    home_win_mask = home_goals_grid > away_goals_grid
    draw_mask = home_goals_grid == away_goals_grid
    away_win_mask = home_goals_grid < away_goals_grid

    draw_strength = float(model["draw_strength"])
    loss_strength = float(model["loss_resistance_strength"])
    win_strength = float(model["win_strength"])
    volatility_strength = float(model["volatility_strength"])
    min_factor = float(model["min_outcome_factor"])
    max_factor = float(model["max_outcome_factor"])

    avg_draw_index = (home_params["draw_index"] + away_params["draw_index"]) / 2.0
    avg_volatility = (home_params["volatility"] + away_params["volatility"]) / 2.0

    draw_factor = 1.0 + draw_strength * (avg_draw_index - 1.0) - volatility_strength * (avg_volatility - 1.0)
    home_win_factor = 1.0 + win_strength * (home_params["win_index"] - 1.0) - loss_strength * (away_params["loss_resistance"] - 1.0)
    away_win_factor = 1.0 + win_strength * (away_params["win_index"] - 1.0) - loss_strength * (home_params["loss_resistance"] - 1.0)

    draw_factor = float(np.clip(draw_factor, min_factor, max_factor))
    home_win_factor = float(np.clip(home_win_factor, min_factor, max_factor))
    away_win_factor = float(np.clip(away_win_factor, min_factor, max_factor))

    adjusted = matrix.copy()
    adjusted[home_win_mask] *= home_win_factor
    adjusted[draw_mask] *= draw_factor
    adjusted[away_win_mask] *= away_win_factor
    adjusted /= adjusted.sum()

    probs = adjusted.ravel()
    home_scores = home_goals_grid.ravel().astype(np.int16)
    away_scores = away_goals_grid.ravel().astype(np.int16)

    raw_home_win = float(matrix[home_win_mask].sum())
    raw_draw = float(matrix[draw_mask].sum())
    raw_away_win = float(matrix[away_win_mask].sum())
    adj_home_win = float(adjusted[home_win_mask].sum())
    adj_draw = float(adjusted[draw_mask].sum())
    adj_away_win = float(adjusted[away_win_mask].sum())

    metadata = {
        "raw_home_win": raw_home_win,
        "raw_draw": raw_draw,
        "raw_away_win": raw_away_win,
        "adj_home_win": adj_home_win,
        "adj_draw": adj_draw,
        "adj_away_win": adj_away_win,
        "home_win_factor": home_win_factor,
        "draw_factor": draw_factor,
        "away_win_factor": away_win_factor,
    }
    return probs, home_scores, away_scores, metadata


def fixture_lambdas(
    fixtures: pd.DataFrame,
    team_params: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    model = config["model"]
    params = team_params.set_index("Team").to_dict("index")
    avg_goals = 1.0
    home_adv = float(model["home_advantage_multiplier"])
    min_lambda = float(model["min_lambda"])
    max_lambda = float(model["max_lambda"])
    mode = str(model.get("fixture_lambda_mode", "recompute"))

    out_rows = []
    for row in fixtures.to_dict("records"):
        home = row["Home"]
        away = row["Away"]

        if mode == "fixture_columns" and "home_lambda" in row and "away_lambda" in row and not pd.isna(row["home_lambda"]):
            home_lambda = float(row["home_lambda"])
            away_lambda = float(row["away_lambda"])
        else:
            home_lambda = avg_goals * home_adv * params[home]["attack_strength"] * params[away]["defence_weakness"]
            away_lambda = avg_goals / home_adv * params[away]["attack_strength"] * params[home]["defence_weakness"]

        home_lambda = float(np.clip(home_lambda, min_lambda, max_lambda))
        away_lambda = float(np.clip(away_lambda, min_lambda, max_lambda))

        out_rows.append({**row, "home_lambda": home_lambda, "away_lambda": away_lambda})

    return pd.DataFrame(out_rows)




def build_target_points_tables(points: np.ndarray, positions: np.ndarray, teams: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_teams = len(teams)

    pooled = pd.DataFrame({
        "Final pts": points.ravel(),
        "Position": positions.ravel(),
    })
    pooled_stats = pooled.groupby("Final pts").agg(
        Samples=("Position", "size"),
        **{
            "Avg position": ("Position", "mean"),
            "1st %": ("Position", lambda x: (x == 1).mean()),
            "Top3 %": ("Position", lambda x: (x <= 3).mean()),
            "Top6 %": ("Position", lambda x: (x <= 6).mean()),
            "12th %": ("Position", lambda x: (x == n_teams).mean()),
        },
    )
    pooled_pos = (
        pooled.pivot_table(index="Final pts", columns="Position", values="Position", aggfunc="size", fill_value=0)
        .reindex(columns=range(1, n_teams + 1), fill_value=0)
        .div(pooled_stats["Samples"], axis=0)
    )
    pooled_pos.columns = [f"Pos {pos} %" for pos in pooled_pos.columns]
    pooled_out = pooled_stats.join(pooled_pos).reset_index().sort_values("Final pts").reset_index(drop=True)

    team_frames: list[pd.DataFrame] = []
    for team_idx, team in enumerate(teams):
        team_df = pd.DataFrame({
            "Final pts": points[:, team_idx],
            "Position": positions[:, team_idx],
        })
        team_stats = team_df.groupby("Final pts").agg(
            Samples=("Position", "size"),
            **{
                "Avg position": ("Position", "mean"),
                "1st %": ("Position", lambda x: (x == 1).mean()),
                "Top3 %": ("Position", lambda x: (x <= 3).mean()),
                "Top6 %": ("Position", lambda x: (x <= 6).mean()),
                "12th %": ("Position", lambda x: (x == n_teams).mean()),
            },
        )
        team_pos = (
            team_df.pivot_table(index="Final pts", columns="Position", values="Position", aggfunc="size", fill_value=0)
            .reindex(columns=range(1, n_teams + 1), fill_value=0)
            .div(team_stats["Samples"], axis=0)
        )
        team_pos.columns = [f"Pos {pos} %" for pos in team_pos.columns]
        team_out = team_stats.join(team_pos).reset_index()
        team_out.insert(0, "Team", team)
        team_frames.append(team_out)

    by_team_out = pd.concat(team_frames, ignore_index=True).sort_values(["Team", "Final pts"]).reset_index(drop=True)
    return pooled_out, by_team_out

def run_simulation(config_path: str | Path = "config.json") -> SimulationResult:
    config_path = Path(config_path)
    base_dir = config_path.parent
    config = load_config(config_path)
    current, fixtures = read_inputs(config, base_dir)

    teams = current["Team"].tolist()
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    n_teams = len(teams)
    n = int(config["simulation_count"])
    rng = np.random.default_rng(int(config.get("random_seed", 0)))

    team_params = compute_team_parameters(current, config)
    param_map = team_params.set_index("Team").to_dict("index")
    fixture_model = fixture_lambdas(fixtures, team_params, config)

    points = np.tile(current["Pts"].to_numpy(dtype=np.int16), (n, 1))
    gf = np.tile(current["GF"].to_numpy(dtype=np.int16), (n, 1))
    ga = np.tile(current["GA"].to_numpy(dtype=np.int16), (n, 1))

    fixture_details = []
    for row in fixture_model.to_dict("records"):
        home = row["Home"]
        away = row["Away"]
        h_idx = team_to_idx[home]
        a_idx = team_to_idx[away]

        probs, home_scores, away_scores, meta = build_score_distribution(
            float(row["home_lambda"]),
            float(row["away_lambda"]),
            param_map[home],
            param_map[away],
            config["model"],
        )
        cumulative = np.cumsum(probs)
        cumulative[-1] = 1.0
        draw = rng.random(n)
        score_idx = np.searchsorted(cumulative, draw, side="right")
        h_goals = home_scores[score_idx]
        a_goals = away_scores[score_idx]

        gf[:, h_idx] += h_goals
        ga[:, h_idx] += a_goals
        gf[:, a_idx] += a_goals
        ga[:, a_idx] += h_goals

        home_wins = h_goals > a_goals
        away_wins = h_goals < a_goals
        draws = ~(home_wins | away_wins)

        points[home_wins, h_idx] += 3
        points[away_wins, a_idx] += 3
        points[draws, h_idx] += 1
        points[draws, a_idx] += 1

        fixture_details.append({**row, **meta})

    gd = gf - ga
    score = points.astype(np.float64) * 1_000_000 + gd.astype(np.float64) * 1_000 + gf.astype(np.float64)
    if config.get("ranking", {}).get("random_tie_breaker", True):
        score += rng.random(score.shape) * 1e-6

    order = np.argsort(-score, axis=1)
    positions = np.empty((n, n_teams), dtype=np.int8)
    position_counts = np.zeros((n_teams, n_teams), dtype=np.int64)
    position_sums = np.zeros(n_teams, dtype=np.float64)

    for pos in range(n_teams):
        teams_at_pos = order[:, pos]
        counts = np.bincount(teams_at_pos, minlength=n_teams)
        position_counts[:, pos] = counts
        positions[np.arange(n), teams_at_pos] = pos + 1
        position_sums += counts * (pos + 1)

    avg_points = points.mean(axis=0)
    avg_gf = gf.mean(axis=0)
    avg_ga = ga.mean(axis=0)
    avg_gd = gd.mean(axis=0)
    avg_position = position_sums / n

    summary_rows = []
    dist_rows = []
    for idx, team in enumerate(teams):
        probs = position_counts[idx] / n
        summary_rows.append(
            {
                "Team": team,
                "Avg pts": avg_points[idx],
                "Avg GF": avg_gf[idx],
                "Avg GA": avg_ga[idx],
                "Avg GD": avg_gd[idx],
                "Avg position": avg_position[idx],
                "1st %": probs[0],
                "Top3 %": probs[:3].sum(),
                "Top6 %": probs[:6].sum(),
                "12th %": probs[-1],
            }
        )
        row = {"Team": team}
        for pos in range(1, n_teams + 1):
            row[str(pos)] = probs[pos - 1]
        dist_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["Avg position", "Avg pts"], ascending=[True, False])
    position_distribution = pd.DataFrame(dist_rows).set_index("Team").loc[summary["Team"]].reset_index()
    target_points, target_points_by_team = build_target_points_tables(points=points, positions=positions, teams=teams)
    fixture_model = pd.DataFrame(fixture_details)

    return SimulationResult(
        summary=summary,
        position_distribution=position_distribution,
        target_points=target_points,
        target_points_by_team=target_points_by_team,
        team_parameters=team_params,
        fixture_model=fixture_model,
    )


def write_excel(result: SimulationResult, config: dict[str, Any], current: pd.DataFrame, fixtures: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.summary.to_excel(writer, sheet_name="Summary", index=False)
        result.position_distribution.to_excel(writer, sheet_name="PositionDistribution", index=False)
        result.target_points.to_excel(writer, sheet_name="TargetPoints", index=False)
        result.target_points_by_team.to_excel(writer, sheet_name="TargetPointsByTeam", index=False)
        current.to_excel(writer, sheet_name="CurrentTable", index=False)
        fixtures.to_excel(writer, sheet_name="Fixtures", index=False)
        result.team_parameters.to_excel(writer, sheet_name="TeamParameters", index=False)
        result.fixture_model.to_excel(writer, sheet_name="FixtureModel", index=False)
        pd.DataFrame([config["model"]]).to_excel(writer, sheet_name="ModelSettings", index=False)

        workbook = writer.book
        for sheet_name in workbook.sheetnames:
            ws = workbook[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
            for column_cells in ws.columns:
                max_len = 0
                col_letter = column_cells[0].column_letter
                for cell in column_cells:
                    value = cell.value
                    if value is not None:
                        max_len = max(max_len, len(str(value)))
                ws.column_dimensions[col_letter].width = min(max_len + 2, 32)

            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    header = ws.cell(row=1, column=cell.column).value
                    if header and (str(header).endswith("%") or str(header).isdigit()):
                        cell.number_format = "0.00%"
                    elif isinstance(cell.value, float):
                        cell.number_format = "0.00"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Veikkausliiga regular-season Monte Carlo simulator")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--simulations", type=int, default=None, help="Override simulation count")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", default=None, help="Override output xlsx path")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.simulations is not None:
        config["simulation_count"] = args.simulations
    if args.seed is not None:
        config["random_seed"] = args.seed
    if args.output is not None:
        config["data"]["output_xlsx"] = args.output

    temp_config_path = config_path
    if args.simulations is not None or args.seed is not None or args.output is not None:
        temp_config_path = config_path.parent / ".runtime_config.json"
        with open(temp_config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)

    result = run_simulation(temp_config_path)
    current, fixtures = read_inputs(config, config_path.parent)
    output_path = Path(config_path.parent) / config["data"]["output_xlsx"]
    write_excel(result, config, current, fixtures, output_path)

    print(f"Wrote {output_path}")
    print(result.summary[["Team", "Avg pts", "Avg position", "1st %", "Top3 %", "Top6 %", "12th %"]].to_string(index=False))

    if temp_config_path.name == ".runtime_config.json":
        temp_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
