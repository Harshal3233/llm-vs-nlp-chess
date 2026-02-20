import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import chess
import chess.pgn
import chess.svg

st.set_page_config(page_title="LLM vs NLP Chess", layout="wide")

SUMMARY_PATH = "data/summary.csv"
RESULTS_PATH = "data/results.jsonl"

@st.cache_data
def load_summary(path):
    return pd.read_csv(path)

@st.cache_data
def load_results(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def build_live_3d(table: pd.DataFrame):
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
        df["winner"],
        df["result"],
        df["llm_color"],
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

    def frame_data(k):
        d = df.iloc[:k+1]
        cd = customdata_all[:k+1]

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
                size=sizes.iloc[:k+1],
                color=d["color"],
                opacity=0.94,
                line=dict(width=2, color=cheat_ring[:k+1]),
            ),
            customdata=cd,
            hovertemplate=hover,
            showlegend=False,
        )
        current = go.Scatter3d(
            x=[d["game"].iloc[-1]], y=[d["cum_score"].iloc[-1]], z=[d["cum_illegal"].iloc[-1]],
            mode="markers",
            marker=dict(size=8, color="rgba(255,255,255,0.92)", opacity=0.9),
            hoverinfo="skip",
            showlegend=False,
        )
        return [glow, line, pts, current]

    frames = [go.Frame(data=frame_data(k), name=str(k)) for k in range(len(df))]
    fig = go.Figure(data=frame_data(0), frames=frames)

    axis_style = dict(
        showbackground=True,
        backgroundcolor="rgb(11,11,16)",
        gridcolor="rgba(255,255,255,0.045)",
        zerolinecolor="rgba(255,255,255,0.07)",
        tickfont=dict(size=11, color="rgba(255,255,255,0.70)"),
        titlefont=dict(size=15, color="rgba(255,255,255,0.92)"),
        showspikes=False,
    )

    fig.update_layout(
        template="plotly_dark",
        height=610,
        showlegend=False,
        scene=dict(
            xaxis=dict(title="Game Index", **axis_style),
            yaxis=dict(title="Cumulative Score (LLM)", **axis_style),
            zaxis=dict(title="Cumulative Illegal Flags", **axis_style),
            bgcolor="rgb(7,7,10)",
            camera=dict(eye=dict(x=1.85, y=1.05, z=0.65), up=dict(x=0, y=0, z=1)),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.02, y=0.01,
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
            x=0.02, y=-0.02,
            xanchor="left", yanchor="top",
            len=0.58,
            currentvalue=dict(prefix="", font=dict(size=10)),
            steps=[dict(
                method="animate",
                args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                label=""
            ) for k in range(len(df))]
        )],
    )
    return fig

def render_board(rec, ply):
    pgn_text = rec.get("pgn")
    if not pgn_text:
        return None, 0
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    moves = list(game.mainline_moves())
    ply = max(0, min(ply, len(moves)))
    b = chess.Board()
    last = None
    for m in moves[:ply]:
        last = m
        b.push(m)
    check_sq = b.king(b.turn) if b.is_check() else None
    svg = chess.svg.board(board=b, size=520, coordinates=True, lastmove=last, check=check_sq)
    return svg, len(moves)

st.title("LLM vs NLP Chess Dashboard")

summary = load_summary(SUMMARY_PATH)
records = load_results(RESULTS_PATH)

st.plotly_chart(build_live_3d(summary), use_container_width=True)
st.dataframe(summary, use_container_width=True, height=320)

st.subheader("Game viewer")
game_idx = st.number_input("Game index", 0, max(0, len(records)-1), 0, 1)
rec = records[int(game_idx)]

if "pgn" in rec and rec.get("pgn"):
    ply = st.slider("Move", 0, len(rec.get("moves", [])), 0)
    svg, _ = render_board(rec, ply)
    st.components.v1.html(svg, height=560, scrolling=False)
else:
    st.info("No PGN found in results.jsonl. If you want the chessboard replay, we’ll export PGN from Colab next.")

st.subheader("Method summary")
st.markdown(
"""
- The LLM agent receives the position (FEN) and the list of legal UCI moves, and must output exactly one move.
- The NLP baseline is a lightweight rule policy (captures/checks + randomness).
- A rules engine validates every move. Illegal proposals are logged as illegal flags, and a legal replacement is played so games remain valid.
- The dashboard reports W/D/L, cumulative score, illegal flags, and latency.
- Results typically show the LLM can generate plausible moves but struggles against consistent rule policies without calibration, and sometimes fails constraints (illegal proposals).
- Calibration improvements: stricter decoding/parsing, self-correction on illegal moves, structured features (checks/captures), opening book for early plies.
"""
)
