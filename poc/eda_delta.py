"""
EDA - Exploratory Data Analysis
Delta Airlines Flight Passenger Prediction
Rapport interactif Plotly - Standard Airbus/Amadeus
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/raw/delta_t100_raw.csv")
REPORT_DIR  = Path("poc/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DELTA_RED   = "#E31837"
DELTA_BLUE  = "#003057"
DELTA_GOLD  = "#C8A96E"
BG_DARK     = "#0D1117"
BG_CARD     = "#161B22"
TEXT_COLOR  = "#E6EDF3"

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("[INFO] Chargement dataset Delta Airlines...")
df = pd.read_csv(DATA_PATH)
print(f"[✓] {len(df):,} enregistrements | {df.shape[1]} variables")

# ─── FEATURE ENGINEERING DE BASE ──────────────────────────────────────────────
df["route"]         = df["origin"] + " → " + df["dest"]
df["year_month"]    = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
df["revenue_est"]   = df["passengers"] * df["avg_ticket_price"]
df["season"] = df["month"].map({
    12:"Winter", 1:"Winter", 2:"Winter",
    3:"Spring",  4:"Spring",  5:"Spring",
    6:"Summer",  7:"Summer",  8:"Summer",
    9:"Fall",   10:"Fall",   11:"Fall"
})

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    title_font=dict(size=16, color=TEXT_COLOR),
    legend=dict(bgcolor=BG_CARD, bordercolor="#30363D", borderwidth=1),
    colorway=[DELTA_RED, DELTA_BLUE, DELTA_GOLD, "#58A6FF", "#3FB950", "#FF7B72"]
)

def apply_template(fig, title=""):
    fig.update_layout(**PLOTLY_TEMPLATE)
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — KPI CARDS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def fig_kpi_overview():
    total_pax   = df["passengers"].sum()
    avg_lf      = df["load_factor"].mean()
    total_rev   = df["revenue_est"].sum()
    n_routes    = df["route"].nunique()
    best_route  = df.groupby("route")["load_factor"].mean().idxmax()
    best_lf     = df.groupby("route")["load_factor"].mean().max()

    fig = go.Figure()
    kpis = [
        ("✈️ Total Passengers", f"{total_pax/1e6:.1f}M", DELTA_RED),
        ("📊 Avg Load Factor",  f"{avg_lf:.1f}%",        DELTA_BLUE),
        ("💰 Est. Revenue",     f"${total_rev/1e9:.1f}B", DELTA_GOLD),
        ("🛫 Routes Analyzed",  str(n_routes),            "#58A6FF"),
        ("🏆 Best Route LF",    f"{best_lf:.1f}%",        "#3FB950"),
    ]
    annotations = []
    for i, (label, value, color) in enumerate(kpis):
        x = 0.1 + i * 0.2
        annotations += [
            dict(x=x, y=0.65, text=value, showarrow=False,
                 font=dict(size=28, color=color, family="Inter"),
                 xref="paper", yref="paper", xanchor="center"),
            dict(x=x, y=0.35, text=label, showarrow=False,
                 font=dict(size=12, color=TEXT_COLOR),
                 xref="paper", yref="paper", xanchor="center"),
        ]
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text="🔴 Delta Air Lines — Fleet Performance Dashboard (2019-2023)",
                   x=0.5, xanchor="center", font=dict(size=20, color=DELTA_RED)),
        annotations=annotations, height=200,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=60, b=20)
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — LOAD FACTOR ÉVOLUTION TEMPORELLE
# ══════════════════════════════════════════════════════════════════════════════
def fig_lf_timeline():
    monthly = df.groupby(["year", "month"]).agg(
        avg_lf=("load_factor", "mean"),
        total_pax=("passengers", "sum")
    ).reset_index()
    monthly["date_label"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    monthly = monthly.sort_values("date_label")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=monthly["date_label"], y=monthly["avg_lf"],
        name="Load Factor %", line=dict(color=DELTA_RED, width=2.5),
        fill="tozeroy", fillcolor="rgba(227,24,55,0.1)"
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=monthly["date_label"], y=monthly["total_pax"],
        name="Passengers", marker_color=DELTA_BLUE, opacity=0.4
    ), secondary_y=True)

    # Annotation COVID
    fig.add_vrect(x0="2020-03-01", x1="2021-06-01",
                  fillcolor="rgba(255,123,114,0.15)",
                  annotation_text="COVID Impact", annotation_position="top left",
                  line_width=0)

    fig.update_yaxes(title_text="Load Factor (%)", secondary_y=False, color=DELTA_RED)
    fig.update_yaxes(title_text="Passengers", secondary_y=True, color=DELTA_BLUE)
    apply_template(fig, "📈 Load Factor & Passenger Volume — Monthly Trend 2019-2023")
    fig.update_layout(height=420)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — HEATMAP SAISONNALITÉ
# ══════════════════════════════════════════════════════════════════════════════
def fig_seasonality_heatmap():
    pivot = df.pivot_table(
        values="load_factor", index="year", columns="month", aggfunc="mean"
    ).round(1)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0,"#003057"],[0.5,"#C8A96E"],[1,"#E31837"]],
        text=pivot.values.round(1),
        texttemplate="%{text}%",
        textfont=dict(size=11),
        hoverongaps=False,
        colorbar=dict(title="Load Factor %", tickcolor=TEXT_COLOR)
    ))
    apply_template(fig, "🌡️ Load Factor Heatmap — Year × Month Seasonality")
    fig.update_layout(height=320)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — TOP ROUTES BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
def fig_top_routes():
    route_stats = df.groupby("route").agg(
        avg_lf=("load_factor", "mean"),
        total_pax=("passengers", "sum"),
        avg_price=("avg_ticket_price", "mean")
    ).reset_index().sort_values("avg_lf", ascending=True).tail(15)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Avg Load Factor by Route",
                                        "Total Passengers by Route"))
    fig.add_trace(go.Bar(
        y=route_stats["route"], x=route_stats["avg_lf"],
        orientation="h", marker_color=DELTA_RED,
        text=route_stats["avg_lf"].round(1).astype(str) + "%",
        textposition="outside", name="Load Factor"
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=route_stats["route"], x=route_stats["total_pax"],
        orientation="h", marker_color=DELTA_BLUE,
        text=(route_stats["total_pax"]/1000).round(0).astype(int).astype(str) + "K",
        textposition="outside", name="Passengers"
    ), row=1, col=2)
    apply_template(fig, "🛫 Top Routes Performance — Delta Air Lines")
    fig.update_layout(height=500, showlegend=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — DISTRIBUTION LOAD FACTOR PAR SAISON
# ══════════════════════════════════════════════════════════════════════════════
def fig_lf_by_season():
    colors = {"Winter": DELTA_BLUE, "Spring": "#3FB950",
               "Summer": DELTA_RED,  "Fall":   DELTA_GOLD}
    fig = go.Figure()
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        data = df[df["season"] == season]["load_factor"]
        fig.add_trace(go.Violin(
            x=[season]*len(data), y=data,
            name=season, box_visible=True,
            meanline_visible=True,
            fillcolor=colors[season],
            opacity=0.7,
            line_color=TEXT_COLOR
        ))
    apply_template(fig, "🌸 Load Factor Distribution by Season")
    fig.update_layout(height=400, violinmode="group")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — SCATTER PRIX vs LOAD FACTOR
# ══════════════════════════════════════════════════════════════════════════════
def fig_price_vs_lf():
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="avg_ticket_price", y="load_factor",
        color="season", size="passengers",
        hover_data=["route", "year", "month"],
        color_discrete_map={
            "Winter": DELTA_BLUE, "Spring": "#3FB950",
            "Summer": DELTA_RED,  "Fall":   DELTA_GOLD
        },
        labels={"avg_ticket_price": "Avg Ticket Price ($)",
                "load_factor": "Load Factor (%)"}
    )
    # Ligne de tendance
    z = np.polyfit(sample["avg_ticket_price"], sample["load_factor"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample["avg_ticket_price"].min(),
                          sample["avg_ticket_price"].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=p(x_line),
        mode="lines", name="Trend",
        line=dict(color="white", width=2, dash="dash")
    ))
    apply_template(fig, "💲 Ticket Price vs Load Factor — Correlation Analysis")
    fig.update_layout(height=420)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — BUBBLE MAP HUBS DELTA
# ══════════════════════════════════════════════════════════════════════════════
def fig_hub_map():
    COORDS = {
        "ATL": (33.64, -84.43), "DTW": (42.21, -83.35),
        "MSP": (44.88, -93.22), "SLC": (40.79, -111.98),
        "SEA": (47.45, -122.31), "BOS": (42.36, -71.01),
        "LGA": (40.78, -73.87), "LAX": (33.94, -118.41),
        "JFK": (40.64, -73.78), "MCO": (28.43, -81.31),
        "MIA": (25.80, -80.29)
    }
    hub_stats = df.groupby("origin").agg(
        avg_lf=("load_factor", "mean"),
        total_pax=("passengers", "sum"),
        n_routes=("dest", "nunique")
    ).reset_index()
    hub_stats["lat"] = hub_stats["origin"].map(lambda x: COORDS.get(x, (0,0))[0])
    hub_stats["lon"] = hub_stats["origin"].map(lambda x: COORDS.get(x, (0,0))[1])
    hub_stats = hub_stats[hub_stats["lat"] != 0]

    fig = go.Figure(go.Scattergeo(
        lat=hub_stats["lat"], lon=hub_stats["lon"],
        text=hub_stats["origin"],
        mode="markers+text",
        textposition="top center",
        marker=dict(
            size=hub_stats["total_pax"] / hub_stats["total_pax"].max() * 50 + 10,
            color=hub_stats["avg_lf"],
            colorscale=[[0, DELTA_BLUE], [0.5, DELTA_GOLD], [1, DELTA_RED]],
            colorbar=dict(title="Avg LF %"),
            line=dict(color="white", width=1)
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Avg LF: %{marker.color:.1f}%<br>"
            "Total Pax: %{customdata:,.0f}<extra></extra>"
        ),
        customdata=hub_stats["total_pax"]
    ))
    fig.update_geos(
        scope="usa", bgcolor=BG_DARK,
        landcolor="#1C2128", coastlinecolor="#30363D",
        showlakes=True, lakecolor=BG_DARK
    )
    apply_template(fig, "🗺️ Delta Air Lines — Hub Network Performance Map USA")
    fig.update_layout(height=480, geo=dict(bgcolor=BG_DARK))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def fig_correlation():
    num_cols = ["load_factor", "avg_ticket_price", "distance", "seats",
                "passengers", "day_of_week", "month", "seasonality_index",
                "covid_impact_factor", "is_holiday_period"]
    corr = df[num_cols].corr().round(3)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, DELTA_BLUE], [0.5, "#0D1117"], [1, DELTA_RED]],
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Correlation", tickcolor=TEXT_COLOR)
    ))
    apply_template(fig, "🔗 Feature Correlation Matrix — Target: Load Factor")
    fig.update_layout(height=480)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — COVID IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def fig_covid_impact():
    yearly = df.groupby("year").agg(
        avg_lf=("load_factor", "mean"),
        total_pax=("passengers", "sum"),
        avg_price=("avg_ticket_price", "mean")
    ).reset_index()

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Avg Load Factor by Year",
                                        "Total Passengers by Year",
                                        "Avg Ticket Price by Year"))
    colors = [DELTA_BLUE, DELTA_RED, "#FF7B72", DELTA_GOLD, "#3FB950"]

    for i, (col, fmt) in enumerate([("avg_lf", ".1f%"), ("total_pax", ","), ("avg_price", "$.0f")]):
        fig.add_trace(go.Bar(
            x=yearly["year"].astype(str), y=yearly[col],
            marker_color=colors, name=col,
            text=yearly[col].apply(
                lambda v: f"{v:.1f}%" if "lf" in col else
                          f"{v/1e6:.1f}M" if "pax" in col else f"${v:.0f}"
            ),
            textposition="outside"
        ), row=1, col=i+1)

    apply_template(fig, "📉 COVID-19 Impact Analysis — Delta Air Lines 2019-2023")
    fig.update_layout(height=380, showlegend=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLAGE RAPPORT HTML
# ══════════════════════════════════════════════════════════════════════════════
def build_html_report():
    print("\n[INFO] Génération des figures...")

    figs = {
        "kpi":        fig_kpi_overview(),
        "timeline":   fig_lf_timeline(),
        "heatmap":    fig_seasonality_heatmap(),
        "routes":     fig_top_routes(),
        "season":     fig_lf_by_season(),
        "price":      fig_price_vs_lf(),
        "map":        fig_hub_map(),
        "corr":       fig_correlation(),
        "covid":      fig_covid_impact(),
    }

    html_parts = []
    for name, fig in figs.items():
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(name == "kpi")))
        print(f"[✓] Figure générée : {name}")

    REPORT_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Delta Air Lines — ML EDA Report 2019-2023</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0D1117; color:#E6EDF3; font-family:'Inter',sans-serif; }}
  .header {{
    background: linear-gradient(135deg, #003057 0%, #E31837 100%);
    padding: 40px; text-align:center;
  }}
  .header h1 {{ font-size:2.2rem; font-weight:700; letter-spacing:1px; }}
  .header p  {{ font-size:1rem; opacity:0.85; margin-top:8px; }}
  .badge {{
    display:inline-block; background:rgba(255,255,255,0.15);
    border-radius:20px; padding:4px 14px; margin:4px;
    font-size:0.8rem; font-weight:600;
  }}
  .section {{
    max-width:1400px; margin:0 auto; padding:20px 24px;
  }}
  .section-title {{
    font-size:1.1rem; font-weight:600; color:#C8A96E;
    border-left:4px solid #E31837; padding-left:12px;
    margin:28px 0 12px;
  }}
  .card {{
    background:#161B22; border:1px solid #30363D;
    border-radius:12px; padding:16px; margin-bottom:20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }}
  .grid-2 {{
    display:grid; grid-template-columns:1fr 1fr; gap:20px;
  }}
  .footer {{
    text-align:center; padding:24px;
    color:#8B949E; font-size:0.85rem;
    border-top:1px solid #30363D; margin-top:40px;
  }}
  @media(max-width:900px) {{ .grid-2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="header">
  <h1>✈️ Delta Air Lines — ML EDA Report</h1>
  <p>Exploratory Data Analysis | Flight Passenger Prediction Platform</p>
  <div style="margin-top:16px">
    <span class="badge">🗓️ 2019–2023</span>
    <span class="badge">✈️ Delta DL Only</span>
    <span class="badge">🛫 20 Routes</span>
    <span class="badge">🤖 ML Ready</span>
    <span class="badge">📊 Plotly Interactive</span>
  </div>
</div>

<div class="section">
  <div class="section-title">📊 Fleet KPI Overview</div>
  <div class="card">{html_parts[0]}</div>

  <div class="section-title">📈 Temporal Analysis</div>
  <div class="card">{html_parts[1]}</div>

  <div class="section-title">🌡️ Seasonality Patterns</div>
  <div class="grid-2">
    <div class="card">{html_parts[2]}</div>
    <div class="card">{html_parts[4]}</div>
  </div>

  <div class="section-title">🛫 Route Performance</div>
  <div class="card">{html_parts[3]}</div>

  <div class="section-title">🗺️ Network Map</div>
  <div class="card">{html_parts[6]}</div>

  <div class="section-title">🔗 Feature Analysis</div>
  <div class="grid-2">
    <div class="card">{html_parts[5]}</div>
    <div class="card">{html_parts[7]}</div>
  </div>

  <div class="section-title">📉 COVID Impact</div>
  <div class="card">{html_parts[8]}</div>
</div>

<div class="footer">
  Delta Air Lines ML Platform — EDA Report generated automatically<br>
  Standard: Airbus/Amadeus MLOps 2026 | Stack: Python · Plotly · Pandas
</div>
</body>
</html>"""

    report_path = REPORT_DIR / "eda_delta_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(REPORT_HTML)

    print(f"\n[✓] Rapport HTML généré : {report_path}")
    print(f"[✓] Taille              : {report_path.stat().st_size / 1024:.0f} KB")
    return report_path


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    report_path = build_html_report()

    print("\n" + "="*60)
    print("EDA SUMMARY — DELTA AIR LINES")
    print("="*60)
    print(f"Records      : {len(df):,}")
    print(f"Avg LF       : {df['load_factor'].mean():.1f}%")
    print(f"Best season  : {df.groupby('season')['load_factor'].mean().idxmax()}")
    print(f"Best route   : {df.groupby('route')['load_factor'].mean().idxmax()}")
    print(f"COVID drop   : {df[df['year']==2019]['load_factor'].mean():.1f}% → "
          f"{df[df['year']==2020]['load_factor'].mean():.1f}%")
    print(f"\n[✓] Ouvre le rapport : {report_path}")
    print("[✓] ÉTAPE 3 COMPLÈTE")