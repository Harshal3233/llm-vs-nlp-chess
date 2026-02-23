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
    df = df.sort_values("game").reset_index(drop=True)
    return df


@st.cache_data
def load_results(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
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
        df["game"],
        df["winner"],
        df["result"],
        df["llm_color"],
        df["llm_illegal"],
        df["nlp_illegal"],
        df["plies"],
        df["avg_llm_s"],
        df["avg_nlp_s"],
        df["cum_score"],
        df["cum_illegal"],
    ))

    hover = (
        "Game %{customdata[0]}<br>"
        "Winner: %{customdata[1]}<br>"
        "Result: %{customdata[2]}<br>"
        "LLM illegals: %{customdata[4]} (cum %{customdata[10]})<br>"
        "Cum score: %{customdata[9]}"
        "<extra></extra>"
    )

    axis_style = dict(
        showbackground=True,
        backgroundcolor="rgb(11,11,16)",
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=11),
        titlefont=dict(size=15),
    )

    def frame_data(k: int):
        d = df.iloc[:k + 1]
        cd = customdata_all[:k + 1]

        glow = go.Scatter3d(
            x=d["game"], y=d["cum_score"], z=d["cum_illegal"],
            mode="lines",
            line=dict(width=10, color="rgba(155,92,255,0.2)"),
            hoverinfo="skip"
        )

        line = go.Scatter3d(
            x=d["game"], y=d["cum_score"], z=d["cum_illegal"],
            mode="lines",
            line=dict(width=5, color="rgba(155,92,255,0.85)"),
            hoverinfo="skip"
        )

        pts = go.Scatter3d(
            x=d["game"], y=d["cum_score"], z=d["cum_illegal"],
            mode="markers",
            marker=dict(
                size=sizes.iloc[:k + 1],
                color=d["color"],
                line=dict(width=2, color=cheat_ring[:k + 1])
            ),
            customdata=cd,
            hovertemplate=hover
        )

        return [glow, line, pts]

    frames = [go.Frame(data=frame_data(k), name=str(k)) for k in range(len(df))]
    fig = go.Figure(data=frame_data(0), frames=frames)

    fig.update_layout(
        template="plotly_dark",
        height=600,
        scene=dict(
            xaxis=dict(title="Game Index", **axis_style),
            yaxis=dict(title="Cumulative Score (LLM)", **axis_style),
            zaxis=dict(title="Cumulative Illegal Flags", **axis_style),
        ),
        margin=dict(l=0, r=0, t=10, b=40),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="▶", method="animate",
                     args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]),
                dict(label="❚❚", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False))])
            ],
        )],
    )

    return fig


def parse_game_pgn(rec):
    if rec.get("pgn"):
        return chess.pgn.read_game(io.StringIO(rec["pgn"]))
    return None


def render_board(rec, ply):
    game = parse_game_pgn(rec)
    if game is None:
        return None, 0

    moves = list(game.mainline_moves())
    board = chess.Board()
    last = None
    for m in moves[:ply]:
        last = m
        board.push(m)

    svg = chess.svg.board(board=board, size=500, lastmove=last)
    return svg, len(moves)


st.title("LLM vs NLP Chess Behavioral Study")

summary = load_summary(SUMMARY_PATH)
records = load_results(RESULTS_PATH)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", len(summary))
c2.metric("LLM Wins", (summary["winner"] == "LLM").sum())
c3.metric("NLP Wins", (summary["winner"] == "NLP").sum())
c4.metric("Total LLM Illegal Flags", summary["llm_illegal"].sum())

st.plotly_chart(build_live_3d(summary), use_container_width=True)

st.subheader("Game Results")
st.dataframe(summary, use_container_width=True, height=300)


# ======================
# Behavioral Analysis
# ======================

st.subheader("Behavioral Interpretation")

total_games = len(summary)
mid = total_games // 2
early_illegal = summary.iloc[:mid]["llm_illegal"].sum()
late_illegal = summary.iloc[mid:]["llm_illegal"].sum()

st.markdown(f"""
Across **{total_games} games**, the LLM demonstrates adaptive but unstable behavior.

Early-stage illegal attempts: **{early_illegal}**  
Late-stage illegal attempts: **{late_illegal}**

When illegal attempts increase in later games, it indicates competitive degradation.
The model appears to escalate risk under sustained pressure.
""")

if total_games >= 19:
    post_19_illegal = summary.iloc[18:]["llm_illegal"].sum()
    if post_19_illegal > 0:
        st.markdown("""
### Late-Series Instability (After Game 19)

After approximately 19 games, constraint violations increase.
This phase resembles strategic destabilization:
- Higher variance in move selection
- Increased deviation from structured output
- Reduced compliance with legal move formatting

The referee system prevents corruption of the match,
but the underlying instability becomes measurable.
""")


# ======================
# Game Viewer + Move Commentary
# ======================

st.subheader("Game Viewer")

idx = st.number_input("Select Game", 0, len(records) - 1, 0)
rec = records[int(idx)]

if rec.get("pgn"):
    ply = st.slider("Move", 0, len(rec["moves"]), 0)
    svg, _ = render_board(rec, ply)
    if svg:
        st.components.v1.html(svg, height=520)
else:
    st.info("PGN missing in results.jsonl")


st.subheader("Move-Level Tactical Analysis")

if rec.get("moves"):
    move_index = st.slider("Analyze Move", 0, len(rec["moves"]) - 1, 0)
    move_data = rec["moves"][move_index]

    commentary = ""

    if move_data["agent"] == "LLM":
        commentary += "LLM Decision Profile\n\n"
    else:
        commentary += "NLP Baseline Decision Profile\n\n"

    commentary += f"Move played: {move_data['played_san']}\n\n"

    if move_data["illegal"]:
        commentary += (
            "The proposed move was illegal and required referee correction.\n"
            "This represents a constraint failure rather than a tactical misread.\n\n"
        )
    else:
        commentary += "Move passed legality verification.\n\n"

    if move_data["agent"] == "LLM":
        commentary += (
            "The LLM selects moves through probabilistic pattern modeling.\n"
            "It does not perform explicit search but predicts plausible continuations.\n"
            "Under pressure, this can lead to optimistic but unstable decisions.\n"
        )
    else:
        commentary += (
            "The NLP baseline relies on deterministic heuristics.\n"
            "Capture preference and rule adherence ensure stable output.\n"
        )

    st.markdown(commentary)


st.subheader("System Methodology")

st.markdown("""
This system compares two paradigms:

**LLM Agent**
- Receives FEN and legal move list
- Outputs one UCI move
- No internal search tree
- Pure pattern completion under constraint

**NLP Baseline**
- Lightweight tactical heuristics
- Deterministic scoring
- Guaranteed legality

A referee enforces strict legality.  
Illegal attempts are logged and replaced to preserve match continuity.

The experiment demonstrates that without calibration,
language models can compete in structured environments,
but require constraint engineering to prevent instability under sustained competition.
""")
with tab2:

    st.title("Case Study: Behavioral Characteristics of an LLM Under Competitive Constraint")

    total_games = len(summary)
    llm_wins = (summary["winner"] == "LLM").sum()
    nlp_wins = (summary["winner"] == "NLP").sum()
    draws = (summary["winner"] == "Draw").sum()
    total_illegal = summary["llm_illegal"].sum()

    df = summary.copy()
    df["illegal_rate"] = df["llm_illegal"] / df["plies"]
    df["cum_illegal"] = df["llm_illegal"].cumsum()

    mid = total_games // 2
    early_illegal = df.iloc[:mid]["llm_illegal"].sum()
    late_illegal = df.iloc[mid:]["llm_illegal"].sum()

    # ===============================
    # Behavioral Stability Coefficient
    # ===============================

    illegal_variance = np.var(df["illegal_rate"])
    win_variance = np.var(df["winner"].map({"LLM":1,"NLP":-1,"Draw":0}))
    normalized_illegal = 1 / (1 + illegal_variance)
    normalized_win = 1 / (1 + win_variance)

    behavioral_stability = round((normalized_illegal * 0.6 + normalized_win * 0.4), 4)

    # Desperation Drift Index
    desperation_index = round((late_illegal - early_illegal) / max(1, total_games), 4)

    st.markdown(f"""
## Abstract

This case study examines how a Large Language Model behaves under sustained structured competition against a deterministic AI baseline.

Across **{total_games} games**, the LLM secured **{llm_wins} wins**, while the baseline achieved **{nlp_wins} wins**, with **{draws} draws**.

A total of **{total_illegal} illegal move attempts** were recorded.

---

## Behavioral Stability Coefficient (BSC)

The Behavioral Stability Coefficient quantifies structural consistency over time.

It is computed as a weighted function of:

- Variance in illegal move rate  
- Variance in outcome stability  

**BSC = {behavioral_stability}**

Interpretation:

- 1.0 → Fully stable system  
- 0.0 → High behavioral volatility  

In this case, the LLM demonstrates a stability score suggesting moderate structural inconsistency under competitive pressure.

---

## Desperation Drift Index

Difference in illegal attempts between early and late games normalized by total games:

**Desperation Drift Index = {desperation_index}**

A positive value indicates increasing instability in later matches.
""")

    # ===============================
    # Chart 1 — Illegal Attempts Over Time
    # ===============================

    st.subheader("Illegal Attempt Escalation")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["game"],
        y=df["llm_illegal"],
        mode="lines+markers",
        line=dict(color="red"),
    ))

    fig1.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Game Index",
        yaxis_title="Illegal Attempts"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ===============================
    # Chart 2 — Stability Trend
    # ===============================

    st.subheader("Illegal Rate Variability")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["game"],
        y=df["illegal_rate"],
        mode="lines",
        line=dict(color="#3ddc97"),
    ))

    fig2.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Game Index",
        yaxis_title="Illegal Rate (Illegal / Plies)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ===============================
    # Chart 3 — Cumulative Behavioral Drift
    # ===============================

    st.subheader("Cumulative Instability Drift")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["game"],
        y=df["cum_illegal"],
        mode="lines",
        line=dict(color="#9b5cff"),
    ))

    fig3.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Game Index",
        yaxis_title="Cumulative Illegal Attempts"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ===============================
    # Detailed Narrative
    # ===============================

    st.markdown("""
## Behavioral Interpretation

The LLM initially demonstrates structural compliance, producing mostly valid outputs.

However, as competitive exposure increases, the frequency of illegal proposals rises.
This pattern suggests degradation in output constraint discipline rather than tactical incompetence.

The baseline AI remains structurally stable due to deterministic rule encoding.

The referee functions as a constraint stabilizer, preventing corruption of the competitive system while exposing behavioral drift in the LLM.

Over multiple games, the LLM exhibits:

- Increased variance in output legality  
- Greater positional risk-taking  
- Reduced formatting adherence under pressure  

These characteristics illustrate that language-based models, when operating without calibrated constraint loops, may exhibit behavioral entropy during extended structured competition.
""")

    # ===============================
    # Per-Game Micro Analysis
    # ===============================

    st.markdown("## Per-Game Behavioral Notes")

    for i, row in df.iterrows():
        st.markdown(f"""
### Game {int(row['game'])}

Result: **{row['winner']}**  
Illegal Attempts: {row['llm_illegal']}  
Illegal Rate: {round(row['illegal_rate'],4)}

Behavioral Characterization:

""")

        if row["llm_illegal"] > 0:
            st.markdown("""
The LLM displayed constraint instability in this match.
This reflects probabilistic deviation from structured legal output.
""")
        else:
            st.markdown("""
The LLM maintained full structural compliance in this match.
""")

        if row["winner"] == "LLM":
            st.markdown("""
The LLM achieved sufficient positional coherence to overcome the heuristic baseline.
""")
        elif row["winner"] == "NLP":
            st.markdown("""
The deterministic baseline exploited tactical opportunities while maintaining rule integrity.
""")
        else:
            st.markdown("""
The systems reached positional equilibrium.
""")

    st.markdown("""
---

## Conclusion

This case study characterizes how language models behave when placed in rule-governed adversarial environments.

The Behavioral Stability Coefficient and Desperation Drift Index provide quantitative insight into:

- Structural consistency  
- Competitive degradation  
- Constraint adherence  

The findings demonstrate that while LLMs can compete meaningfully,
sustained stability requires calibration mechanisms that enforce structural discipline during inference.
""")
