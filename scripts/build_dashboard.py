"""Build the interactive dashboard (docs/index.html).

Reads the cleaned data in data/ and the notebook results in results/results.json,
and writes a single self-contained HTML file that GitHub Pages can serve.

    python scripts/build_dashboard.py

Run ipl_inference_and_prediction.ipynb first so results/results.json exists.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "dashboard" / "template.html"
RESULTS = ROOT / "results" / "results.json"
OUT = ROOT / "docs" / "index.html"

TEAM_FIX = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# (substring found in the raw venue name, shared key, label shown on the dashboard)
VENUES = [
    ("Chidambaram", "Chidambaram", "MA Chidambaram, Chennai"),
    ("Wankhede", "Wankhede", "Wankhede, Mumbai"),
    ("Eden Gardens", "Eden Gardens", "Eden Gardens, Kolkata"),
    ("Chinnaswamy", "Chinnaswamy", "Chinnaswamy, Bengaluru"),
    ("Feroz Shah", "Feroz Shah", "Arun Jaitley (Kotla), Delhi"),
    ("Arun Jaitley", "Feroz Shah", "Arun Jaitley (Kotla), Delhi"),
    ("Rajiv Gandhi", "Rajiv Gandhi", "Rajiv Gandhi, Hyderabad"),
    ("Sawai Mansingh", "Sawai Mansingh", "Sawai Mansingh, Jaipur"),
    ("Punjab Cricket", "Punjab Cricket", "PCA Stadium, Mohali"),
    ("Mohali", "Punjab Cricket", "PCA Stadium, Mohali"),
    ("Narendra Modi", "Narendra Modi", "Narendra Modi, Ahmedabad"),
    ("Motera", "Narendra Modi", "Narendra Modi, Ahmedabad"),
    ("Dubai", "Dubai", "Dubai International"),
    ("Sharjah", "Sharjah", "Sharjah"),
    ("Zayed", "Zayed", "Sheikh Zayed, Abu Dhabi"),
    ("Brabourne", "Brabourne", "Brabourne, Mumbai"),
    ("DY Patil", "DY Patil", "DY Patil, Navi Mumbai"),
    ("Maharashtra Cricket", "Maharashtra Cricket", "MCA Stadium, Pune"),
]


def venue_label(name):
    if isinstance(name, str):
        for key, _, label in VENUES:
            if key in name:
                return label
    return "Other venues"


def clean(df):
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()
    return df


def main():
    if not RESULTS.exists():
        raise SystemExit("results/results.json not found. Run ipl_inference_and_prediction.ipynb first.")

    matches = clean(pd.read_csv(ROOT / "data" / "matches_2008-2024.csv")).rename(columns={"id": "match_id"})
    balls = clean(pd.read_csv(ROOT / "data" / "deliveries_2008-2024.csv.gz"))
    for col in ["team1", "team2", "toss_winner", "winner"]:
        matches[col] = matches[col].replace(TEAM_FIX)
    balls = balls[balls["inning"] <= 2].merge(matches[["match_id", "season"]], on="match_id")

    teams = sorted(set(matches.team1) | set(matches.team2))
    tix = {t: i for i, t in enumerate(teams)}
    venues = sorted({venue_label(v) for v in matches.venue})
    vix = {v: i for i, v in enumerate(venues)}

    first_innings = balls.groupby(["match_id", "inning"]).total_runs.sum().unstack()[1]
    sixes = balls[balls.batsman_runs == 6].groupby("match_id").size()
    rows = []
    for m in matches.itertuples():
        rows.append([
            int(m.season), tix[m.team1], tix[m.team2], tix[m.winner] if pd.notna(m.winner) else -1,
            tix[m.toss_winner], 1 if m.toss_decision == "field" else 0, vix[venue_label(m.venue)],
            int(first_innings[m.match_id]), int(sixes.get(m.match_id, 0)), 1 if m.match_type == "Final" else 0,
        ])

    balls["phase"] = pd.cut(balls["over"], [-1, 5, 14, 19], labels=["pp", "mid", "death"])
    legal = ~balls["extras_type"].isin(["wides", "noballs"])
    runs = balls.groupby(["season", "phase"], observed=True).total_runs.sum()
    overs = balls[legal].groupby(["season", "phase"], observed=True).size() / 6
    phase = {int(s): {p: [int(runs[(s, p)]), round(float(overs[(s, p)]), 3)] for p in ["pp", "mid", "death"]}
             for s in sorted(balls.season.unique())}

    def by_season(series, minimum):
        out = {}
        for (player, season), value in series.items():
            out.setdefault(player, {})[int(season)] = int(value)
        return {p: v for p, v in out.items() if sum(v.values()) >= minimum}

    batters = by_season(balls.groupby(["batter", "season"]).batsman_runs.sum(), 100)
    not_bowler = ["run out", "retired hurt", "retired out", "obstructing the field"]
    credited = balls[(balls.is_wicket == 1) & (~balls.dismissal_kind.isin(not_bowler))]
    bowlers = by_season(credited.groupby(["bowler", "season"]).size(), 5)

    data = dict(teams=teams, venues=venues, matches=rows, phase=phase, batters=batters, bowlers=bowlers)
    data.update(json.loads(RESULTS.read_text()))
    payload = json.dumps(data, separators=(",", ":"))
    assert "</script" not in payload

    body = TEMPLATE.read_text().replace("/*__DATA__*/", payload)
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
        + body + "\n</body>\n</html>\n"
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
