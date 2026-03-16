"""
AI Crowd Safety Monitor - Dashboard V2.0
Author: Mrityunjay Singh
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import time
from sklearn.metrics import confusion_matrix, classification_report

from final_pipeline import (
    extract_motion_features,
    label_motion_data,
    prepare_temporal_data,
    train_and_predict,
    frame_to_timestamp,
    VIDEO_PATH,
    FPS,
)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Crowd Safety Monitor",
    page_icon="🛡️",
    layout="wide",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Top banner */
    .top-banner {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 18px 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }
    .banner-title { font-size: 1.6rem; font-weight: 700; color: #e0e0e0; margin: 0; }
    .banner-subtitle { font-size: 0.85rem; color: #7a8aaa; margin: 4px 0 0 0; }
    .status-pill {
        background: #0d2b1a;
        border: 1px solid #1a6b3a;
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.85rem;
        color: #2ecc71;
        font-weight: 600;
    }

    /* KPI cards */
    .kpi-card {
        background: #1a1f2e;
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 22px 20px;
        height: 100%;
    }
    .kpi-label { font-size: 0.75rem; color: #7a8aaa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 0; line-height: 1; }
    .kpi-sub   { font-size: 0.8rem; color: #7a8aaa; margin-top: 6px; }

    /* Status badge */
    .badge-low      { color: #2ecc71; }
    .badge-medium   { color: #f39c12; }
    .badge-high     { color: #e74c3c; }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 28px 0 14px 0;
        border-left: 3px solid #3b82f6;
        padding-left: 10px;
    }

    /* Event log table */
    .event-row {
        background: #1a1f2e;
        border: 1px solid #2a3550;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .event-time  { font-size: 0.85rem; color: #7a8aaa; }
    .event-desc  { font-size: 0.9rem;  color: #e0e0e0; font-weight: 500; }
    .event-badge {
        background: #2d1515;
        border: 1px solid #e74c3c;
        color: #e74c3c;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Trust metric bars */
    .trust-label { font-size: 0.82rem; color: #a0aec0; margin-bottom: 4px; }
    .trust-value { font-size: 1.1rem; font-weight: 700; color: #e0e0e0; }

    /* Divider */
    hr { border-color: #2a3550; margin: 24px 0; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #2a3550;
    }
    [data-testid="stSidebar"] .stMarkdown p { color: #a0aec0; }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── HELPER: frame → "HH:MM:SS" string ─────────────────────────────────────────
def frame_to_timestamp_str(frame_id, fps):
    total_seconds = int(frame_id / fps)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── PIPELINE (cached so it only runs once per video) ───────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline():
    """
    Cached: feature extraction + labelling + model training run exactly once.
    Subsequent reloads (filter changes, sidebar interaction) return instantly.
    """
    df = extract_motion_features(VIDEO_PATH)
    df = label_motion_data(df)
    df = prepare_temporal_data(df)
    y_true, y_pred, test_df = train_and_predict(df)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    test_df = test_df.copy()
    test_df["prediction"]     = y_pred
    test_df["timestamp"]      = test_df["frame_id"].apply(lambda x: frame_to_timestamp(x, FPS))
    test_df["timestamp_str"]  = test_df["frame_id"].apply(lambda x: frame_to_timestamp_str(x, FPS))

    # Attach a real datetime for the slider widget
    _base = datetime.now().replace(second=0, microsecond=0)
    test_df["datetime"] = test_df["frame_id"].apply(
        lambda x: _base + timedelta(seconds=int(x / FPS))
    )

    return y_true, y_pred, test_df, report, cm


# ── SKELETON LOADING SCREEN ───────────────────────────────────────────────────
st.markdown("""
<style>
  @keyframes shimmer {
    0%   { background-position: -800px 0; }
    100% { background-position:  800px 0; }
  }
  .skeleton-wrap {
    background: #0f1117;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
  }
  .skeleton-card {
    background: #1a1f2e;
    border: 1px solid #2a3550;
    border-radius: 14px;
    padding: 32px 36px;
    width: 100%;
    max-width: 640px;
    margin-bottom: 16px;
  }
  .skeleton-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e0e0e0;
    margin: 0 0 6px 0;
  }
  .skeleton-sub {
    font-size: 0.82rem;
    color: #3a4560;
    margin: 0 0 28px 0;
  }
  .stage-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 18px;
  }
  .stage-icon {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
  }
  .stage-icon.done    { background:#0d2b1a; color:#2ecc71; border:1px solid #1a6b3a; }
  .stage-icon.active  { background:#0d1f3a; color:#3b82f6; border:1px solid #1e4a8a;
                        animation: pulse-ring 1.2s ease-in-out infinite; }
  .stage-icon.waiting { background:#1a1f2e; color:#3a4560; border:1px solid #2a3550; }
  @keyframes pulse-ring {
    0%,100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.4); }
    50%      { box-shadow: 0 0 0 6px rgba(59,130,246,0);  }
  }
  .stage-text-label { font-size:0.9rem; font-weight:600; color:#e0e0e0; }
  .stage-text-sub   { font-size:0.75rem; color:#7a8aaa; margin-top:2px; }
  .progress-track {
    background: #2a3550;
    border-radius: 8px;
    height: 6px;
    margin-top: 24px;
    overflow: hidden;
  }
  .progress-fill {
    height: 6px;
    border-radius: 8px;
    background: linear-gradient(90deg, #1e4a8a, #3b82f6, #60a5fa);
    background-size: 800px 6px;
    animation: shimmer 1.6s linear infinite;
  }
  .skeleton-block {
    border-radius: 6px;
    background: linear-gradient(90deg, #1a1f2e 25%, #232a3d 50%, #1a1f2e 75%);
    background-size: 800px 100%;
    animation: shimmer 1.6s linear infinite;
  }
  .skeleton-kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    width: 100%;
    max-width: 640px;
  }
  .skeleton-kpi {
    background: #1a1f2e;
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 18px 14px;
  }
</style>
""", unsafe_allow_html=True)

loading_placeholder = st.empty()

def show_skeleton_pct(pct: int):
    """Render skeleton with an arbitrary fill percentage (0-100)."""
    stages = [
        ("📹  Reading the Video",
         "The AI is scanning every frame of the footage, measuring how fast people are moving",
         "&#9678;"),
        ("🏷️  Deciding What's Normal",
         "Comparing each moment against typical crowd behaviour to spot anything unusual",
         "&#8857;"),
        ("🤖  Running the Safety Check",
         "The AI model is making its final call — safe or unsafe — for every second of video",
         "&#8859;"),
    ]
    if pct < 33:
        stage = 0
    elif pct < 66:
        stage = 1
    else:
        stage = 2

    rows_html = ""
    for i, (label, sub, icon) in enumerate(stages):
        if i < stage:
            css_cls, ico = "done",    "&#10003;"
        elif i == stage:
            css_cls, ico = "active",  icon
        else:
            css_cls, ico = "waiting", icon
        rows_html = (rows_html
            + '<div class="stage-row">'
            + '<div class="stage-icon ' + css_cls + '">' + ico + '</div>'
            + '<div>'
            + '<p class="stage-text-label">' + label + '</p>'
            + '<p class="stage-text-sub">'   + sub   + '</p>'
            + '</div></div>'
        )

    skeleton_blocks = ""
    for h, w in [(14, "60%"), (10, "80%"), (10, "45%"), (10, "70%")]:
        skeleton_blocks += ('<div class="skeleton-block" style="height:'
                            + str(h) + 'px; width:' + w
                            + '; margin-bottom:10px;"></div>')
    kpi_blocks = ""
    for _ in range(4):
        kpi_blocks += ('<div class="skeleton-kpi">'
                       '<div class="skeleton-block" style="height:10px;width:55%;margin-bottom:10px;"></div>'
                       '<div class="skeleton-block" style="height:28px;width:70%;"></div>'
                       '</div>')

    html = (
        '<div class="skeleton-wrap">'
        '<div class="skeleton-card">'
        '<p class="skeleton-title">AI Crowd Safety Monitor</p>'
        '<p class="skeleton-sub">Analysing the footage &#8212; this usually takes about 40 seconds</p>'
        + rows_html +
        '<div class="progress-track">'
        '<div class="progress-fill" style="width:' + str(pct) + '%;"></div>'
        '</div></div>'
        '<div class="skeleton-kpi-row">' + kpi_blocks + '</div>'
        '<div class="skeleton-card" style="margin-top:16px;">' + skeleton_blocks + '</div>'
        '</div>'
    )
    loading_placeholder.markdown(html, unsafe_allow_html=True)


# ── LOAD WITH SKELETON ─────────────────────────────────────────────────────────
if "pipeline_done" not in st.session_state:

    # ── Smooth laminar-flow progress ──────────────────────────────────────────
    # We generate a dense sequence of (pct, sleep) pairs that follows an
    # ease-in-out sine curve so the bar never jumps — it glides from 0 → 90
    # before the real pipeline fires, then snaps cleanly to 100 afterward.
    #
    # Strategy:
    #   • 0 → 90  over ~120 micro-steps using a sine ease-in-out envelope
    #   • Each step moves only 0.75 pct on average — visually continuous
    #   • The sleep between steps is modulated: slower in the middle (0.04 s),
    #     slightly quicker at both ends (0.02 s) — mimics laminar deceleration
    #   • No abrupt jumps; pct is always a float rounded to 1 dp

    import math

    def _smooth_phases(start: float, end: float, steps: int):
        """Yield (pct, delay_s) with sine ease-in-out between start and end."""
        for i in range(steps + 1):
            t = i / steps                                    # 0.0 → 1.0
            ease = (1 - math.cos(math.pi * t)) / 2          # sine ease-in-out
            pct  = start + (end - start) * ease
            # Delay: slowest at the midpoint (t≈0.5), fastest at edges
            #        range 0.018 s (edges) … 0.048 s (middle)
            delay = 0.018 + 0.030 * math.sin(math.pi * t)
            yield round(pct, 1), delay

    for pct, delay in _smooth_phases(0, 90, 120):
        show_skeleton_pct(int(pct))
        time.sleep(delay)

    y_true, y_pred, test_df, report, cm = run_pipeline()

    # Smooth 90 → 100 in a quick glide after pipeline completes
    for pct, delay in _smooth_phases(90, 100, 20):
        show_skeleton_pct(int(pct))
        time.sleep(delay * 0.4)   # fast finish

    time.sleep(0.25)
    st.session_state["pipeline_done"] = (y_true, y_pred, test_df, report, cm)
else:
    y_true, y_pred, test_df, report, cm = st.session_state["pipeline_done"]

loading_placeholder.empty()


# ── DERIVE SUMMARY STATS ───────────────────────────────────────────────────────
p85 = test_df["mean_motion"].quantile(0.85)
p60 = test_df["mean_motion"].quantile(0.60)

latest_motion = test_df["mean_motion"].iloc[-1]
latest_pred   = test_df["prediction"].iloc[-1]

transitions  = np.diff(np.concatenate([[0], test_df["prediction"].values]))
event_starts = np.where(transitions == 1)[0]
total_events = len(event_starts)

# Build structured event list (with peak intensity)
events = []
in_event, s_frame, s_time, peak = False, None, None, 0.0
for _, row in test_df.iterrows():
    pred = row["prediction"]
    idx  = row["frame_id"]
    ts   = row["timestamp"]
    mot  = row["mean_motion"]
    if pred == 1 and not in_event:
        in_event, s_frame, s_time, peak = True, idx, ts, mot
    elif pred == 1 and in_event:
        peak = max(peak, mot)
    elif pred == 0 and in_event:
        e_frame  = idx - 1
        duration = round((e_frame - s_frame) / FPS, 1)
        events.append({
            "start": s_time,
            "end": frame_to_timestamp(e_frame, FPS),
            "frames": f"{s_frame}–{e_frame}",
            "duration_s": duration,
            "peak_intensity": round(peak, 4),
        })
        in_event = False
if in_event:
    e_frame  = test_df["frame_id"].iloc[-1]
    duration = round((e_frame - s_frame) / FPS, 1)
    events.append({
        "start": s_time,
        "end": frame_to_timestamp(e_frame, FPS),
        "frames": f"{s_frame}–{e_frame}",
        "duration_s": duration,
        "peak_intensity": round(peak, 4),
    })

total_alert_secs = sum(e["duration_s"] for e in events)

precision_pct = report.get("1.0", {}).get("precision", 0) * 100
recall_pct    = report.get("1.0", {}).get("recall",    0) * 100
f1_pct        = report.get("1.0", {}).get("f1-score",  0) * 100
accuracy_pct  = report.get("accuracy", 0) * 100

tn, fp, fn, tp = cm.ravel()
false_alert_rate = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ AI Crowd Safety Monitor")
    st.markdown(f"**Monitoring:** `{VIDEO_PATH}`")
    st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
    st.markdown("**Status:** System Active 🟢")
    st.markdown("---")

    st.markdown("### What do you want to see?")

    # ── Risk Level Filter
    selected_risk_levels = st.multiselect(
        "Show moments that were...",
        options=["Normal", "Abnormal"],
        default=["Normal", "Abnormal"],
        help="'Normal' = calm crowd, 'Abnormal' = AI flagged it as dangerous",
    )
    selected_risk_values = [0 if lvl == "Normal" else 1 for lvl in selected_risk_levels]

    # ── Time Range Filter
    min_dt = test_df["datetime"].min().to_pydatetime()
    max_dt = test_df["datetime"].max().to_pydatetime()

    selected_time_range = st.slider(
        "Zoom into a time window",
        min_value=min_dt,
        max_value=max_dt,
        value=(min_dt, max_dt),
        format="HH:mm:ss",
        help="Drag the handles to focus on a specific part of the video",
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#3a4560;'>Your filters update the chart, map, and event log below.</p>",
        unsafe_allow_html=True,
    )

# ── APPLY FILTERS ──────────────────────────────────────────────────────────────
filtered_df = test_df[
    (test_df["prediction"].isin(selected_risk_values)) &
    (test_df["datetime"] >= selected_time_range[0]) &
    (test_df["datetime"] <= selected_time_range[1])
].copy()


# ── TOP BANNER ─────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%d %b %Y  •  %H:%M:%S")
st.markdown(f"""
<div class="top-banner">
  <div>
    <p class="banner-title">AI Crowd Safety Monitor</p>
    <p class="banner-subtitle">UMN Crowd Activity Dataset  •  Offline Analysis  •  {now}</p>
  </div>
  <div class="status-pill">● System Active</div>
</div>
""", unsafe_allow_html=True)


# ── KPI ROW ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">At a Glance &#8212; What is happening right now?</p>', unsafe_allow_html=True)

if latest_pred == 1:
    status_class, status_text, status_icon = "badge-high",    "HIGH RISK",     "⚠"
elif latest_motion >= p60:
    status_class, status_text, status_icon = "badge-medium",  "ELEVATED RISK", "◉"
else:
    status_class, status_text, status_icon = "badge-low",     "LOW RISK",      "✔"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
      <p class="kpi-label">Is the crowd safe right now?</p>
      <p class="kpi-value {status_class}">{status_icon} {status_text}</p>
      <p class="kpi-sub">Based on the last moment analysed in the video</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
      <p class="kpi-label">How fast is the crowd moving?</p>
      <p class="kpi-value" style="color:#3b82f6;">{latest_motion:.3f}</p>
      <p class="kpi-sub">Anything above {p85:.3f} is considered dangerously fast</p>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
      <p class="kpi-label">How many danger moments were spotted?</p>
      <p class="kpi-value {'badge-high' if total_events > 0 else 'badge-low'}">{total_events}</p>
      <p class="kpi-sub">Each "event" is a continuous burst of unsafe crowd movement</p>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card">
      <p class="kpi-label">Total time the crowd was at risk</p>
      <p class="kpi-value" style="color:#a78bfa;">{total_alert_secs:.1f}s</p>
      <p class="kpi-sub">Combined seconds where the AI flagged danger</p>
    </div>""", unsafe_allow_html=True)


# ── MOTION TREND CHART (filtered, timestamp axis + tooltips) ──────────────────
st.markdown('<p class="section-header">How did crowd movement change over time?</p>', unsafe_allow_html=True)
st.caption("The line shows how fast the crowd was moving at each moment. Green zone = calm, orange = getting busy, red = danger. The dashed red line is your alert threshold — anything above it triggered a warning.")

if filtered_df.empty:
    st.info("No data matches the current filter selection.")
else:
    chart_df = filtered_df[["frame_id", "mean_motion", "timestamp", "timestamp_str"]].copy()

    max_motion = chart_df["mean_motion"].max() * 1.1 if not chart_df.empty else p85 * 2

    band_df = pd.DataFrame([
        {"y1": 0,    "y2": p60,       "zone": "Normal"},
        {"y1": p60,  "y2": p85,       "zone": "Elevated"},
        {"y1": p85,  "y2": max_motion, "zone": "High Risk"},
    ])

    bands = alt.Chart(band_df).mark_rect(opacity=0.12).encode(
        y=alt.Y("y1:Q"),
        y2=alt.Y2("y2:Q"),
        color=alt.Color("zone:N", scale=alt.Scale(
            domain=["Normal", "Elevated", "High Risk"],
            range=["#2ecc71", "#f39c12", "#e74c3c"]
        ), legend=alt.Legend(title="Risk Zone", orient="top-right",
                             labelColor="#a0aec0", titleColor="#a0aec0"))
    )

    line = alt.Chart(chart_df).mark_line(strokeWidth=1.8, color="#3b82f6").encode(
        x=alt.X("timestamp:N",
                title="Time (HH:MM:SS)",
                axis=alt.Axis(labelColor="#7a8aaa", titleColor="#7a8aaa",
                              labelAngle=-30, labelLimit=80,
                              # show every Nth label to avoid crowding
                              labelOverlap="parity")),
        y=alt.Y("mean_motion:Q", title="Motion Intensity",
                axis=alt.Axis(labelColor="#7a8aaa", titleColor="#7a8aaa")),
        tooltip=[
            alt.Tooltip("timestamp:N",    title="Time in video"),
            alt.Tooltip("frame_id:Q",     title="Frame number"),
            alt.Tooltip("mean_motion:Q",  title="Crowd speed (higher = more chaotic)", format=".4f"),
        ]
    )

    threshold_line = alt.Chart(pd.DataFrame({"y": [p85]})).mark_rule(
        color="#e74c3c", strokeDash=[6, 3], strokeWidth=1.5
    ).encode(y="y:Q")

    final_chart = (bands + line + threshold_line).properties(
        height=280,
        background="#1a1f2e",
        padding={"left": 10, "right": 10, "top": 10, "bottom": 10}
    ).configure_view(strokeWidth=0)

    st.altair_chart(final_chart, use_container_width=True)


# ── INTERACTIVE MAP ────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Where in the area were people at risk?</p>', unsafe_allow_html=True)
st.caption("Each dot represents a moment in the crowd footage, mapped to its approximate location. 🟢 Green dots = everything was calm. 🔴 Red dots = the AI flagged dangerous movement at that spot and time. (Demo uses estimated GPS coordinates — a real deployment would use camera location data.)")

if filtered_df.empty:
    st.info("No data matches the current filter selection.")
else:
    # Mock GPS — plausible coordinates around a stadium/plaza
    np.random.seed(42)
    map_data = filtered_df.copy()
    map_data["lat"] = 44.9740 + np.random.normal(loc=0.0005, scale=0.0012, size=len(map_data))
    map_data["lon"] = -93.2277 + np.random.normal(loc=-0.0008, scale=0.0015, size=len(map_data))

    map_chart = alt.Chart(map_data).mark_circle(size=55, opacity=0.75).encode(
        latitude="lat:Q",
        longitude="lon:Q",
        color=alt.Color(
            "prediction:N",
            scale=alt.Scale(domain=[0, 1], range=["#2ecc71", "#e74c3c"]),
            legend=alt.Legend(
                title="Risk Level",
                labelExpr="datum.value == 1 ? 'Abnormal' : 'Normal'",
                labelColor="#a0aec0",
                titleColor="#a0aec0",
            ),
        ),
        tooltip=[
            alt.Tooltip("timestamp:N",   title="Time in video"),
            alt.Tooltip("prediction:N",  title="Status (0 = Safe, 1 = Danger)"),
            alt.Tooltip("mean_motion:Q", title="Crowd speed", format=".4f"),
        ],
    ).properties(
        title=alt.TitleParams(
            "Crowd Location & Risk Levels",
            color="#a0aec0",
        ),
        height=340,
        background="#1a1f2e",
        padding={"left": 10, "right": 10, "top": 10, "bottom": 10},
    ).project(
        type="mercator"
    ).configure_view(strokeWidth=0)

    st.altair_chart(map_chart, use_container_width=True)


# ── EVENT LOG + TRUST METRICS ──────────────────────────────────────────────────
col_log, col_trust = st.columns([3, 2], gap="large")

with col_log:
    st.markdown('<p class="section-header">When exactly did danger occur?</p>', unsafe_allow_html=True)
    st.caption("Each entry below is a separate window of time when the AI detected dangerous crowd movement. Think of it as a highlight reel of the risky moments.")

    # Filter events that overlap with the selected time range & risk filter
    visible_events = [
        ev for ev in events
        if (1 in selected_risk_values)   # only show if Abnormal is selected
    ] if (1 in selected_risk_values) else []

    if not visible_events:
        st.info("No danger moments match your current filter. Try selecting 'Abnormal' in the sidebar.")
    else:
        for i, ev in enumerate(visible_events, 1):
            st.markdown(f"""
            <div class="event-row">
              <div>
                <p class="event-desc">⚠️ Dangerous crowd movement detected</p>
                <p class="event-time">
                  Started at {ev['start']}, ended at {ev['end']} &nbsp;•&nbsp; Frames {ev['frames']}
                </p>
              </div>
              <div style="display:flex; gap:14px; align-items:center; flex-wrap:wrap;">
                <div style="text-align:right;">
                  <p style="color:#7a8aaa; font-size:0.75rem; margin:0;">Lasted for</p>
                  <p style="color:#a78bfa; font-size:0.9rem; font-weight:600; margin:0;">{ev['duration_s']}s</p>
                </div>
                <div style="text-align:right;">
                  <p style="color:#7a8aaa; font-size:0.75rem; margin:0;">Peak crowd speed</p>
                  <p style="color:#f39c12; font-size:0.9rem; font-weight:600; margin:0;">{ev['peak_intensity']}</p>
                </div>
                <span class="event-badge">DANGER</span>
              </div>
            </div>""", unsafe_allow_html=True)

with col_trust:
    st.markdown('<p class="section-header">Can you trust this AI?</p>', unsafe_allow_html=True)
    st.caption("These numbers tell you how reliable the system is — like a report card for the AI.")

    metrics = [
        ("When it says danger, is it right?",
         "Out of every 100 alerts, this many were real threats — not false alarms.",
         precision_pct, "#3b82f6"),
        ("Does it catch every real danger?",
         "Out of every 100 real danger moments in the video, this many were actually caught.",
         recall_pct,    "#2ecc71"),
        ("Overall grade",
         "A balanced score combining both — how reliable the AI is, overall.",
         f1_pct,        "#a78bfa"),
    ]

    for label, caption, value, color in metrics:
        st.markdown(f"""
        <div style="margin-bottom:18px;">
          <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <span class="trust-label">{label}</span>
            <span class="trust-value" style="color:{color};">{value:.1f}%</span>
          </div>
          <div style="background:#2a3550; border-radius:6px; height:8px;">
            <div style="background:{color}; width:{value:.1f}%; height:8px; border-radius:6px;"></div>
          </div>
          <p style="font-size:0.75rem; color:#7a8aaa; margin-top:4px;">{caption}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:4px;">
      <div class="kpi-card" style="padding:14px;">
        <p class="kpi-label">Correctly caught</p>
        <p class="kpi-value badge-low" style="font-size:1.6rem;">{tp}</p>
        <p class="kpi-sub">Real danger moments the AI spotted ✅</p>
      </div>
      <div class="kpi-card" style="padding:14px;">
        <p class="kpi-label">Missed dangers</p>
        <p class="kpi-value badge-high" style="font-size:1.6rem;">{fn}</p>
        <p class="kpi-sub">Real dangers the AI didn't catch ❌</p>
      </div>
      <div class="kpi-card" style="padding:14px;">
        <p class="kpi-label">False alarms</p>
        <p class="kpi-value badge-medium" style="font-size:1.6rem;">{fp}</p>
        <p class="kpi-sub">Safe moments the AI misread as danger ⚠️</p>
      </div>
      <div class="kpi-card" style="padding:14px;">
        <p class="kpi-label">False alarm rate</p>
        <p class="kpi-value badge-medium" style="font-size:1.6rem;">{false_alert_rate:.1f}%</p>
        <p class="kpi-sub">How often it cries wolf — lower is better</p>
      </div>
    </div>""", unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#3a4560; font-size:0.78rem;">
  AI Crowd Safety Monitor &nbsp;•&nbsp; Prototype / Proof of Concept &nbsp;•&nbsp;
  Crowd-level analysis only — no individual tracking &nbsp;•&nbsp; Mrityunjay Singh
</p>""", unsafe_allow_html=True)