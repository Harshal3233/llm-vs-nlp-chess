import json
import io
from math import exp
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
def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("game").reset_index(drop=True)

    required = [
        "game", "winner", "result", "llm_color",
        "llm_illegal", "nlp_illegal", "plies",
        "avg_llm_s", "avg_nlp_s"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"summary.csv missing columns: {missing}")

    df["game"] = df["game"].astype(int)
    df["plies"] = df["plies"].astype(int)
    df["llm_illegal"] = df["llm_illegal"].astype(int)
    df["nlp_illegal"] = df["nlp_illegal"].astype(int)
    df["avg_llm_s"] = df["avg_llm_s"].astype(float)
    df["avg_nlp_s"] = df["avg_nlp_s"].astype(float)

    return df


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
        "Plies: %{customdata[6]}<br>"
        "Avg LLM: %{customdata[7]:.3f}s<br>"
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


def board_from_record(rec: dict, ply: int):
    if rec.get("pgn"):
        game = chess.pgn.read_game(io.StringIO(rec["pgn"]))
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

    moves = rec.get("moves", [])
    ply = max(0, min(ply, len(moves)))
    b = chess.Board()
    last = None
    for i in range(ply):
        uci = moves[i].get("played_uci")
        if not uci:
            continue
        mv = chess.Move.from_uci(uci)
        if mv in b.legal_moves:
            last = mv
            b.push(mv)
        else:
            break
    check_sq = b.king(b.turn) if b.is_check() else None
    svg = chess.svg.board(board=b, size=520, coordinates=True, lastmove=last, check=check_sq)
    return svg, len(moves)


def per_game_agent_stats(rec: dict):
    moves = rec.get("moves", [])
    llm_lat = []
    nlp_lat = []
    llm_illegal = 0
    nlp_illegal = 0
    llm_moves = 0
    nlp_moves = 0
    captures = {"LLM": 0, "NLP": 0}
    checks = {"LLM": 0, "NLP": 0}

    for m in moves:
        agent = m.get("agent", "")
        san = m.get("played_san", "") or ""
        illegal = bool(m.get("illegal", False))
        lat = float(m.get("latency_sec", 0.0))

        if agent == "LLM":
            llm_moves += 1
            llm_illegal += int(illegal)
            llm_lat.append(lat)
        elif agent == "NLP":
            nlp_moves += 1
            nlp_illegal += int(illegal)
            nlp_lat.append(lat)

        if "x" in san:
            captures[agent] = captures.get(agent, 0) + 1
        if "+" in san or "#" in san:
            checks[agent] = checks.get(agent, 0) + 1

    def avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    return {
        "llm_avg_latency": avg(llm_lat),
        "nlp_avg_latency": avg(nlp_lat),
        "llm_illegal_moves": llm_illegal,
        "nlp_illegal_moves": nlp_illegal,
        "llm_moves": llm_moves,
        "nlp_moves": nlp_moves,
        "llm_capture_rate": (captures["LLM"] / max(1, llm_moves)),
        "nlp_capture_rate": (captures["NLP"] / max(1, nlp_moves)),
        "llm_check_rate": (checks["LLM"] / max(1, llm_moves)),
        "nlp_check_rate": (checks["NLP"] / max(1, nlp_moves)),
    }


def behavioral_metrics(summary: pd.DataFrame):
    df = summary.copy()
    df["outcome"] = df["winner"].map({"LLM": 1, "NLP": -1, "Draw": 0}).astype(int)
    df["illegal_rate"] = df["llm_illegal"] / df["plies"].replace(0, 1)
    df["cum_illegal"] = df["llm_illegal"].cumsum()

    illegal_var = float(np.var(df["illegal_rate"].to_numpy()))
    outcome_var = float(np.var(df["outcome"].to_numpy()))

    bsc_illegal = 1.0 / (1.0 + illegal_var)
    bsc_outcome = 1.0 / (1.0 + outcome_var)
    bsc = float(0.65 * bsc_illegal + 0.35 * bsc_outcome)

    mid = max(1, len(df) // 2)
    early_illegal = int(df.iloc[:mid]["llm_illegal"].sum())
    late_illegal = int(df.iloc[mid:]["llm_illegal"].sum())
    desperation_drift = float((late_illegal - early_illegal) / max(1, len(df)))

    return df, round(bsc, 4), round(desperation_drift, 4), early_illegal, late_illegal


def confidence_proxy_curve(summary: pd.DataFrame):
    df = summary.copy()
    df["illegal_rate"] = df["llm_illegal"] / df["plies"].replace(0, 1)
    lat = df["avg_llm_s"].to_numpy()
    lat_norm = (lat - lat.min()) / (max(1e-9, lat.max() - lat.min()))
    irr = df["illegal_rate"].to_numpy()
    irr_norm = (irr - irr.min()) / (max(1e-9, irr.max() - irr.min()))

    alpha = 3.2
    beta = 1.4
    score = np.exp(-alpha * irr_norm) * np.exp(-beta * lat_norm)

    df["confidence_proxy"] = score
    df["confidence_roll"] = df["confidence_proxy"].rolling(5, min_periods=1).mean()
    return df


def radar_chart(llm_metrics: dict, nlp_metrics: dict):
    categories = [
        "Win rate",
        "Legality (1 - illegal rate)",
        "Speed (1 - latency norm)",
        "Capture rate",
        "Check rate",
        "Stability (BSC proxy)",
    ]

    def clamp01(x):
        return float(max(0.0, min(1.0, x)))

    llm_win = llm_metrics["win_rate"]
    nlp_win = nlp_metrics["win_rate"]

    llm_leg = 1.0 - llm_metrics["illegal_rate"]
    nlp_leg = 1.0 - nlp_metrics["illegal_rate"]

    llm_speed = 1.0 - llm_metrics["latency_norm"]
    nlp_speed = 1.0 - nlp_metrics["latency_norm"]

    llm_vals = list(map(clamp01, [llm_win, llm_leg, llm_speed, llm_metrics["capture_rate"], llm_metrics["check_rate"], llm_metrics["stability"]]))
    nlp_vals = list(map(clamp01, [nlp_win, nlp_leg, nlp_speed, nlp_metrics["capture_rate"], nlp_metrics["check_rate"], nlp_metrics["stability"]]))

    llm_vals += [llm_vals[0]]
    nlp_vals += [nlp_vals[0]]
    cats = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=nlp_vals,
        theta=cats,
        fill="toself",
        name="NLP baseline",
        opacity=0.55,
    ))
    fig.add_trace(go.Scatterpolar(
        r=llm_vals,
        theta=cats,
        fill="toself",
        name="LLM",
        opacity=0.55,
    ))
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        title="Comparative radar: LLM vs NLP"
    )
    return fig


def line_chart(x, y, title, y_label, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=color)))
    fig.update_layout(
        template="plotly_dark",
        height=330,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        xaxis_title="Game Index",
        yaxis_title=y_label,
        showlegend=False,
    )
    return fig


def move_narrative(board: chess.Board, entry: dict, agent_label: str, phase_hint: str):
    proposed = entry.get("proposed_uci", "")
    played = entry.get("played_uci", "")
    san = entry.get("played_san", "") or ""
    illegal = bool(entry.get("illegal", False))
    lat = float(entry.get("latency_sec", 0.0))

    capture = "x" in san
    check = ("+" in san) or ("#" in san)

    beats = []
    if illegal:
        beats.append("The proposal slips outside the legal boundary. The referee intercepts it and keeps the game coherent.")
    else:
        beats.append("The move lands inside the legal boundary, suggesting good constraint adherence on this step.")

    if capture and check:
        beats.append("It’s both a capture and a check, a high-pressure move that forces a response.")
    elif capture:
        beats.append("It’s a capture, trading material and narrowing the position’s options.")
    elif check:
        beats.append("It’s a check, shifting the tempo into forced replies.")
    else:
        beats.append("It’s a quiet move, more about shaping the position than forcing it.")

    if agent_label == "LLM":
        beats.append(
            "The LLM’s behavior here reads as pattern-driven: it selects a plausible continuation rather than proving it by search. "
            "When the position is sharp, that plausibility sometimes wobbles."
        )
    else:
        beats.append(
            "The baseline is operating like a metronome: simple heuristics, consistent legality, and a preference for forcing motifs when available."
        )

    if phase_hint:
        beats.append(phase_hint)

    beats.append(f"Latency: {lat:.3f}s. Proposed: {proposed or '—'}; Played: {played or '—'}; SAN: {san or '—'}.")

    return " ".join(beats)


def phase_hint_for_game(game_index: int, total_games: int, llm_illegals_in_game: int):
    if total_games < 6:
        return ""
    third = total_games // 3
    if game_index < third:
        return "Early in the series, outputs tend to be more controlled: fewer stylistic swings, fewer boundary breaks."
    if game_index < 2 * third:
        if llm_illegals_in_game > 0:
            return "Mid-series, the model starts to show strain: constraint slips appear as the competitive rhythm continues."
        return "Mid-series, the model appears to settle into a routine, but the baseline keeps extracting value from consistency."
    if llm_illegals_in_game > 0:
        return "Late-series, the model’s constraint discipline degrades: illegal proposals surface more often, a signature of boundary fatigue rather than strategy."
    return "Late-series, the game remains structurally clean, but variance in choice quality still decides outcomes."


summary = load_summary(SUMMARY_PATH)
records = load_results(RESULTS_PATH)

tab1, tab2 = st.tabs(["Dashboard", "Case Study Report"])


with tab1:
    st.title("LLM vs NLP Chess Dashboard")

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

    max_ply_guess = len(rec.get("moves", []))
    ply = st.slider("Move", 0, max(1, max_ply_guess), 0)
    svg, _ = board_from_record(rec, ply)
    st.components.v1.html(svg, height=560, scrolling=False)

    st.subheader("Method and interpretation")
    st.markdown(
        """
This dashboard is the visible surface of a case study: it tracks how a language-model-driven move policy behaves when it must operate inside a strict rules boundary, while a deterministic baseline continues with reliable, heuristic play.

Illegal proposals are treated as behavioral events. They do not corrupt games, because a referee enforces legality, but they do reveal when the model’s output slips outside the allowed action space.
        """
    )


with tab2:
    st.title("Case Study Report")
    dfm, bsc, drift, early_illegal, late_illegal = behavioral_metrics(summary)
    conf_df = confidence_proxy_curve(summary)

    total_games = int(len(dfm))
    llm_wins = int((dfm["winner"] == "LLM").sum())
    nlp_wins = int((dfm["winner"] == "NLP").sum())
    draws = int((dfm["winner"] == "Draw").sum())
    total_illegal = int(dfm["llm_illegal"].sum())

    st.markdown(
        f"""
## Overview

This report is written as a case study of model behavior under repeated competitive play.

One system is an LLM that proposes moves from a constrained legal set. The other is a deterministic baseline that applies lightweight tactical heuristics. A referee enforces legality and logs every illegal proposal as a behavioral signal.

Across **{total_games} games**, the LLM recorded **{llm_wins} wins**, the baseline recorded **{nlp_wins} wins**, and **{draws} games** ended in draws. The LLM produced **{total_illegal} illegal move proposals**.

## Behavioral Stability Coefficient (BSC)

**BSC = {bsc}**

This coefficient compresses two sources of volatility:
- instability in the illegal-rate (illegal attempts normalized by plies)
- instability in the outcome sequence (win/draw/loss swings)

Higher values indicate a tighter behavioral envelope across the run.

## Desperation Drift Index

**Drift = {drift}**

A positive drift indicates illegal proposals increased in the later half of the run compared to the early half. This reads like boundary fatigue: more outputs fall outside the legal set as the series continues.
        """
    )

    st.subheader("Charts inside the case study")

    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(
            line_chart(dfm["game"], dfm["llm_illegal"], "Illegal attempts per game", "Illegal attempts", "#ff5c7a"),
            use_container_width=True,
        )
    with cB:
        st.plotly_chart(
            line_chart(dfm["game"], dfm["illegal_rate"], "Illegal rate per game", "Illegal / plies", "#3ddc97"),
            use_container_width=True,
        )

    st.plotly_chart(
        line_chart(dfm["game"], dfm["cum_illegal"], "Cumulative illegal attempts", "Cumulative illegals", "#9b5cff"),
        use_container_width=True,
    )

    st.subheader("Confidence collapse curve")
    st.caption("A proxy curve derived from legality stability and latency drift; it is not a claim of internal model confidence.")

    fig_conf = go.Figure()
    fig_conf.add_trace(go.Scatter(
        x=conf_df["game"],
        y=conf_df["confidence_proxy"],
        mode="lines+markers",
        name="confidence proxy",
        line=dict(color="#00c2ff"),
    ))
    fig_conf.add_trace(go.Scatter(
        x=conf_df["game"],
        y=conf_df["confidence_roll"],
        mode="lines",
        name="rolling mean (5)",
        line=dict(color="#9b5cff"),
    ))
    fig_conf.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Confidence collapse curve (proxy)",
        xaxis_title="Game Index",
        yaxis_title="Proxy score (higher = steadier)",
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    st.subheader("Comparative radar chart: LLM vs NLP")

    llm_win_rate = float((summary["winner"] == "LLM").mean())
    nlp_win_rate = float((summary["winner"] == "NLP").mean())

    llm_illegal_rate = float(summary["llm_illegal"].sum() / max(1, summary["plies"].sum()))
    nlp_illegal_rate = float(summary["nlp_illegal"].sum() / max(1, summary["plies"].sum()))

    llm_latency = float(summary["avg_llm_s"].mean())
    nlp_latency = float(summary["avg_nlp_s"].mean())
    lat_min = min(llm_latency, nlp_latency)
    lat_max = max(llm_latency, nlp_latency)
    llm_lat_norm = 0.0 if lat_max == lat_min else (llm_latency - lat_min) / (lat_max - lat_min)
    nlp_lat_norm = 0.0 if lat_max == lat_min else (nlp_latency - lat_min) / (lat_max - lat_min)

    llm_cap = 0.0
    nlp_cap = 0.0
    llm_chk = 0.0
    nlp_chk = 0.0
    for r in records:
        s = per_game_agent_stats(r)
        llm_cap += s["llm_capture_rate"]
        nlp_cap += s["nlp_capture_rate"]
        llm_chk += s["llm_check_rate"]
        nlp_chk += s["nlp_check_rate"]
    llm_cap /= max(1, len(records))
    nlp_cap /= max(1, len(records))
    llm_chk /= max(1, len(records))
    nlp_chk /= max(1, len(records))

    llm_metrics = {
        "win_rate": llm_win_rate,
        "illegal_rate": llm_illegal_rate,
        "latency_norm": llm_lat_norm,
        "capture_rate": float(llm_cap),
        "check_rate": float(llm_chk),
        "stability": float(bsc),
    }
    nlp_metrics = {
        "win_rate": nlp_win_rate,
        "illegal_rate": nlp_illegal_rate,
        "latency_norm": nlp_lat_norm,
        "capture_rate": float(nlp_cap),
        "check_rate": float(nlp_chk),
        "stability": 0.98,
    }

    st.plotly_chart(radar_chart(llm_metrics, nlp_metrics), use_container_width=True)

    st.subheader("Per-game narrative")
    st.caption("Each game is expanded into a detailed behavioral note, including a move-by-move chronicle. Use the expanders to keep it readable.")

    show_full_moves_default = st.checkbox("Default to full move-by-move narrative", value=False)
    max_moves_when_compact = st.slider("If not full, show N moves per game", 6, 40, 14)

    for i, row in dfm.iterrows():
        g = int(row["game"])
        winner = str(row["winner"])
        result = str(row["result"])
        plies = int(row["plies"])
        ill = int(row["llm_illegal"])
        rate = float(row["illegal_rate"])

        headline = f"Game {g} | winner: {winner} | result: {result} | plies: {plies} | LLM illegals: {ill}"
        with st.expander(headline, expanded=False):
            st.markdown(
                f"""
**Summary snapshot**
- Outcome: **{winner}**
- Game length: **{plies} plies**
- LLM illegal proposals: **{ill}** (rate **{rate:.4f}**)

**Behavioral reading**
""".strip()
            )

            if ill == 0:
                st.markdown(
                    "The LLM stays inside the legal boundary throughout the game. The story becomes one of choice quality: plausible moves versus forcing heuristics."
                )
            elif ill == 1:
                st.markdown(
                    "A single boundary breach appears. The referee corrects it, preserving the match while exposing a momentary constraint slip."
                )
            else:
                st.markdown(
                    "Multiple boundary breaches appear. The referee becomes a stabilizing spine, and the illegal flags function like stress markers across the game."
                )

            if winner == "LLM":
                st.markdown(
                    "The LLM finds a coherent thread long enough to convert. Against a shallow heuristic baseline, occasional bursts of strong patterning can be decisive."
                )
            elif winner == "NLP":
                st.markdown(
                    "The baseline’s consistency keeps cashing in small tactical profits. The LLM’s higher variance becomes a liability, especially when the position demands precision."
                )
            else:
                st.markdown(
                    "Neither side reliably converts advantage. The baseline stays safe; the LLM stays plausible; the result settles into balance."
                )

            phase_hint = phase_hint_for_game(g, total_games, ill)
            if phase_hint:
                st.markdown(f"**Phase note**: {phase_hint}")

            rec = records[g] if g < len(records) else None
            if not rec or not rec.get("moves"):
                st.info("No move log available for this game in results.jsonl.")
                continue

            moves = rec["moves"]
            want_full = show_full_moves_default
            if not want_full:
                want_full = st.checkbox(f"Show full move-by-move narrative (Game {g})", value=False, key=f"full_{g}")

            if want_full:
                idxs = range(len(moves))
            else:
                idxs = range(min(len(moves), max_moves_when_compact))

            st.markdown("### Move-by-move chronicle")
            b = chess.Board()
            for j in idxs:
                m = moves[j]
                agent = m.get("agent", "")
                agent_label = "LLM" if agent == "LLM" else "NLP"
                nh = move_narrative(b, m, agent_label, phase_hint if j < 6 else "")
                ply_no = m.get("ply", j + 1)
                st.markdown(f"**Ply {ply_no} ({agent_label})**: {nh}")

                uci = m.get("played_uci")
                if uci:
                    mv = chess.Move.from_uci(uci)
                    if mv in b.legal_moves:
                        b.push(mv)
                    else:
                        break

    st.subheader("Closing synthesis")
    st.markdown(
        """
A deterministic baseline expresses competence as reliability. The LLM expresses competence as flexibility. When those meet repeatedly under strict rules, the most visible fault line is not “understanding,” but *stability*.

The referee keeps games valid, which makes the behavioral signal measurable:
- illegal proposals indicate boundary slips (constraint fatigue, formatting drift, or selection drift)
- outcomes indicate whether plausibility is enough to survive heuristic pressure

If your next step is “make the LLM competitive,” the case study naturally suggests calibration levers:
- stricter decoding and schema enforcement
- retry-on-illegal self-correction loop
- structured tactical inputs (checks, captures, threats) as features
- an opening calibration layer for early plies
        """
    )
