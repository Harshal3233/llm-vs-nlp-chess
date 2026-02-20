import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import chess
import chess.pgn
import chess.svg

st.set_page_config(page_title="LLM vs NLP Chess Dashboard", layout="wide")

SUMMARY_PATH = "data/summary.csv"
RESULTS_PATH = "data/results.jsonl"


@st.cache_data
def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {
        "llm_illegal": "llm_illegal",
        "nlp_illegal": "nlp_illegal",
        "avg_llm_s": "avg_llm_s",
        "avg_nlp_s": "avg_nlp_s",
        "llm_color": "llm_color",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    required = ["game", "winner", "result", "llm_color", "llm_illegal", "nlp_illegal", "plies", "avg_llm_s", "avg_nlp_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"summary.csv missing columns: {missing}")

    df["game"] = df["game"].astype(int)
    df["plies"] = df["plies"].astype(int)
    df["llm_illegal"] = df["llm_illegal"].astype(int)
    df["nlp_illegal"] = df["nlp_illegal"].astype(int)
    df["avg_llm_s"] = df["avg_llm_s"].astype(float)
    df["avg_nlp_s"] = df["avg_nlp_s"].astype(float)

    return df.sort_values("game").reset_index(drop=True)


@st.cache_data
def load_results(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_live_3d(table: pd.DataFrame) -> go.Figure:
    df = table.copy()
    df["outcome"] = df["winner"].map({"LLM": 1, "NLP": -1, "Draw": 0}).astype(int)
    df["cum_score"] = df["outcome"].cumsum()
    df["cum_illegal"] = df["llm_illegal"].cumsum()

    color_map = {1: "#3ddc97", -1: "#ff5c7a", 0: "#00c2ff"}
    df["color"] = df["outcome"].map(color_map)

    sizes = (5 + (df["plies"] / 75).clip(0, 6)).astype(float)
    cheat_ring = np.where(df["llm_illegal"] > 0, "rgba(255,0,0,0.92)", "rgba(0,0,0,0)")

    customdata_all = list(zip(
        df["game"].astype(int),
        df["winner"].astype(str),
        df["result"].astype(str),
        df["llm_color"].astype(str),
        df["llm_illegal"].astype(int),
        df["nlp_illegal"].astype(int),
        df["plies"].astype(int),
        df["avg_llm_s"].astype(float),
        df["avg_nlp_s"].astype(float),
        df["cum_score"].astype(int),
        df["cum_illegal"].astype(int),
    ))

    hover = (
        "Game %{customdata[0]}<br>"
        "Winner: %{customdata[1]}<br>"
        "Result: %{customdata[2]}<br>"
        "LLM color: %{customdata[3]}<br>"
        "LLM illegals: %{customdata[4]} (cum %{customdata[10]})<br>"
        "NLP illegals: %{customdata[5]}<br>"
        "Plies: %{customdata[6]}<br>"
        "Avg LLM: %{customdata[7]:.3f}s<br>"
        "Avg NLP: %{customdata[8]:.3f}s<br>"
        "Cum score: %{customdata[9]}"
        "<extra></extra>"
    )

    axis_style = dict(
        showbackground=True,
        backgroundcolor="rgb(11,11,16)",
        gridcolor="rgba(255,255,255,0.045)",
        zerolinecolor="rgba(255,255,255,0.07)",
        tickfont=dict(size=11, color="rgba(255,255,255,0.70)"),
        titlefont=dict(size=15, color="rgba(255,255,255,0.92)"),
        showspikes=False,
    )

    def frame_data(k: int):
        d = df.iloc[:k + 1]
        cd = customdata_all[:k + 1]

        glow = go.Scatter3d(
            x=d["game"], y=d["cum_score"], z=d["cum_illegal"],
            mode="lines",
            line=dict(width=11, color="rgba(155,92,255,0.20)"),
            hoverinfo="skip",
            showlegend=False,
        )
        line = go.Scatter3d(
            x=d["game"], y=d["cum_score"], z=d["cum_illegal"],
            mode="lines",
            line=dict(width=5, color="rgba(155,92,255,0.88)"),
            hoverinfo="skip",
            showlegend=False,
        )
        pts = go.Scatter3d(
            x=d["game"], y=d["cum_score"], z=d["cum_illegal"],
            mode="markers",
            marker=dict(
                size=sizes.iloc[:k + 1],
                color=d["color"],
                opacity=0.94,
                line=dict(width=2, color=cheat_ring[:k + 1]),
            ),
            customdata=cd,
            hovertemplate=hover,
            showlegend=False,
        )
        current = go.Scatter3d(
            x=[d["game"].iloc[-1]],
            y=[d["cum_score"].iloc[-1]],
            z=[d["cum_illegal"].iloc[-1]],
            mode="markers",
            marker=dict(size=8, color="rgba(255,255,255,0.92)", opacity=0.9),
            hoverinfo="skip",
            showlegend=False,
        )
        return [glow, line, pts, current]

    frames = [go.Frame(data=frame_data(k), name=str(k)) for k in range(len(df))]
    fig = go.Figure(data=frame_data(0), frames=frames)

    fig.update_layout(
        template="plotly_dark",
        height=610,
        showlegend=False,
        scene=dict(
            xaxis=dict(title="Game Index", **axis_style),
            yaxis=dict(title="Cumulative Score (LLM)", **axis_style),
            zaxis=dict(title="Cumulative Illegal Flags (LLM)", **axis_style),
            bgcolor="rgb(7,7,10)",
            camera=dict(eye=dict(x=1.85, y=1.05, z=0.65), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=10, b=60),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.02, y=0.02,
            xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="▶", method="animate",
                     args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="❚❚", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
            pad=dict(r=4, t=4),
            showactive=False
        )],
        sliders=[dict(
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            len=0.58,
            pad=dict(t=20, b=0),
            currentvalue=dict(prefix="", font=dict(size=10)),
            steps=[dict(
                method="animate",
                args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                label=""
            ) for k in range(len(df))]
        )],
    )

    return fig


def parse_game_pgn(rec: dict):
    pgn_text = rec.get("pgn")
    if not pgn_text:
        return None
    return chess.pgn.read_game(io.StringIO(pgn_text))


def render_board(rec: dict, ply: int):
    game = parse_game_pgn(rec)
    if game is None:
        return None, 0

    moves = list(game.mainline_moves())
    ply = max(0, min(ply, len(moves)))

    board = chess.Board()
    last = None
    for m in moves[:ply]:
        last = m
        board.push(m)

    check_sq = board.king(board.turn) if board.is_check() else None
    svg = chess.svg.board(board=board, size=520, coordinates=True, lastmove=last, check=check_sq)
    return svg, len(moves)


st.title("LLM vs NLP Chess Dashboard")

summary = load_summary(SUMMARY_PATH)
records = load_results(RESULTS_PATH)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", int(len(summary)))
c2.metric("LLM wins", int((summary["winner"] == "LLM").sum()))
c3.metric("NLP wins", int((summary["winner"] == "NLP").sum()))
c4.metric("Draws", int((summary["winner"] == "Draw").sum()))

st.plotly_chart(build_live_3d(summary), use_container_width=True)

st.subheader("Game results")
st.dataframe(summary, use_container_width=True, height=320)

st.subheader("Game viewer")
idx = st.number_input("Game index", min_value=0, max_value=max(0, len(records) - 1), value=0, step=1)
rec = records[int(idx)]

if rec.get("pgn"):
    ply = st.slider("Move", 0, max(1, len(rec.get("moves", []))), 0)
    svg, _ = render_board(rec, ply)
    if svg:
        st.components.v1.html(svg, height=560, scrolling=False)
else:
    st.info("No PGN found in results.jsonl. Regenerate results with PGN if you want board replay.")

st.subheader("Method and interpretation")
st.markdown(
"""
### Setup
This experiment compares two move-selection policies under a strict rules referee.

- **LLM agent**: receives the current position (FEN) and a list of legal UCI moves, then outputs a single UCI move.
- **NLP baseline**: a lightweight rule-driven policy that prefers captures/checks and injects randomness.

### Referee
A rules engine validates every proposed move. If a proposed move is illegal or malformed, the attempt is logged as an illegal flag and a legal replacement move is applied so the game remains valid chess.

### Metrics
- **Winner / result**: win-loss-draw outcome with alternating colors.
- **Cumulative score**: (LLM wins − LLM losses) over the sequence of games.
- **Illegal flags**: frequency of illegal proposals by the LLM.
- **Latency**: average seconds per move for each agent.

### Interpretation
Without calibration, the LLM often produces plausible but inconsistent moves, and sometimes fails the strict output constraint. The baseline remains stable and legal, which typically yields better results. With calibration, the LLM can become competitive against this level of baseline:
- stricter parsing and constrained decoding
- retry-on-illegal self-correction loop
- structured tactical hints (checks/captures) as inputs
- opening calibration for early plies
"""
)
