"""
Feature Engineering - Multi-Level
Delta Airlines Flight Passenger Prediction
Niveaux : Vol / Route / Aéroport / Réseau / Temporel
Standard : Airbus/Amadeus MLOps 2026
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RAW_PATH      = Path("data/raw/delta_t100_raw.csv")
FEATURES_DIR  = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FEATURE ENGINEERING — DELTA AIRLINES")
print("Standard : Airbus/Amadeus MLOps 2026")
print("=" * 60)

df = pd.read_csv(RAW_PATH)
print(f"\n[✓] Dataset chargé : {len(df):,} lignes | {df.shape[1]} colonnes")

# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 1 — FEATURES TEMPORELLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Features Temporelles...")

df["is_weekend"]        = (df["day_of_week"] >= 6).astype(int)
df["is_monday"]         = (df["day_of_week"] == 1).astype(int)
df["is_friday"]         = (df["day_of_week"] == 5).astype(int)
df["quarter"]           = ((df["month"] - 1) // 3 + 1)
df["is_q4"]             = (df["quarter"] == 4).astype(int)
df["is_summer"]         = df["month"].isin([6, 7, 8]).astype(int)
df["is_winter"]         = df["month"].isin([12, 1, 2]).astype(int)
df["is_spring"]         = df["month"].isin([3, 4, 5]).astype(int)
df["is_fall"]           = df["month"].isin([9, 10, 11]).astype(int)
df["month_sin"]         = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]         = np.cos(2 * np.pi * df["month"] / 12)
df["dow_sin"]           = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]           = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Périodes clés aviation US
df["is_thanksgiving"]   = ((df["month"] == 11) & (df["is_holiday_period"] == 1)).astype(int)
df["is_xmas_newyear"]   = ((df["month"] == 12) & (df["is_holiday_period"] == 1)).astype(int)
df["is_july4"]          = ((df["month"] == 7) & (df["is_holiday_period"] == 1)).astype(int)
df["is_peak_travel"]    = (df["month"].isin([6, 7, 8, 11, 12])).astype(int)

print(f"[✓] +{14} features temporelles ajoutées")

# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 2 — FEATURES VOL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Features Vol...")

df["route"]             = df["origin"] + "_" + df["dest"]
df["seats_category"]    = pd.cut(df["seats"],
                                  bins=[0, 100, 150, 200, 999],
                                  labels=["small", "medium", "large", "widebody"])
df["price_per_mile"]    = df["avg_ticket_price"] / df["distance"].clip(lower=1)
df["revenue_per_seat"]  = (df["passengers"] * df["avg_ticket_price"]) / df["seats"].clip(lower=1)
df["yield_metric"]      = df["avg_ticket_price"] / df["distance"].clip(lower=1) * 100
df["is_long_haul"]      = (df["distance"] >= 1500).astype(int)
df["is_medium_haul"]    = ((df["distance"] >= 500) & (df["distance"] < 1500)).astype(int)
df["is_short_haul"]     = (df["distance"] < 500).astype(int)

# Distance binned
df["distance_bin"] = pd.cut(df["distance"],
                              bins=[0, 500, 1000, 1500, 2000, 9999],
                              labels=["<500", "500-1k", "1k-1.5k", "1.5k-2k", ">2k"])

print(f"[✓] +{9} features vol ajoutées")

# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 3 — FEATURES ROUTE (agrégations historiques)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Features Route (agrégations historiques)...")

route_stats = df.groupby("route").agg(
    route_avg_lf        = ("load_factor",       "mean"),
    route_std_lf        = ("load_factor",       "std"),
    route_max_lf        = ("load_factor",       "max"),
    route_min_lf        = ("load_factor",       "min"),
    route_avg_price     = ("avg_ticket_price",  "mean"),
    route_avg_distance  = ("distance",          "mean"),
    route_total_pax     = ("passengers",        "sum"),
    route_n_flights     = ("departures_performed", "sum"),
    route_avg_seats     = ("seats",             "mean"),
).reset_index()

route_stats["route_lf_cv"]          = (route_stats["route_std_lf"] /
                                        route_stats["route_avg_lf"]).round(4)
route_stats["route_popularity_rank"] = route_stats["route_total_pax"].rank(
    ascending=False).astype(int)

df = df.merge(route_stats, on="route", how="left")
print(f"[✓] +{len(route_stats.columns)-1} features route ajoutées")

# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 4 — FEATURES AÉROPORT (hub importance)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Features Aéroport...")

DELTA_HUB_TIER = {
    "ATL": 1, "DTW": 1, "MSP": 1, "SLC": 2,
    "SEA": 2, "BOS": 2, "LGA": 2, "LAX": 2,
    "JFK": 2, "MCO": 3, "MIA": 3
}

airport_stats = df.groupby("origin").agg(
    airport_avg_lf      = ("load_factor",      "mean"),
    airport_total_pax   = ("passengers",       "sum"),
    airport_n_routes    = ("dest",             "nunique"),
    airport_avg_price   = ("avg_ticket_price", "mean"),
).reset_index().rename(columns={"origin": "airport"})

# Origin airport features
df["origin_hub_tier"]       = df["origin"].map(DELTA_HUB_TIER).fillna(3)
df["dest_hub_tier"]         = df["dest"].map(DELTA_HUB_TIER).fillna(3)
df["hub_to_hub"]            = ((df["origin_hub_tier"] == 1) &
                                (df["dest_hub_tier"] == 1)).astype(int)
df["hub_to_spoke"]          = ((df["origin_hub_tier"] == 1) &
                                (df["dest_hub_tier"] >= 2)).astype(int)
df["is_atl_flight"]         = ((df["origin"] == "ATL") |
                                (df["dest"] == "ATL")).astype(int)

df = df.merge(airport_stats.rename(columns={
    "airport": "origin",
    "airport_avg_lf":    "origin_avg_lf",
    "airport_total_pax": "origin_total_pax",
    "airport_n_routes":  "origin_n_routes",
    "airport_avg_price": "origin_avg_price"
}), on="origin", how="left")

print(f"[✓] +{6} features aéroport ajoutées")

# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 5 — FEATURES RÉSEAU / COMPAGNIE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Features Réseau & Compagnie...")

network_monthly = df.groupby(["year", "month"]).agg(
    network_avg_lf      = ("load_factor",      "mean"),
    network_total_pax   = ("passengers",       "sum"),
    network_avg_price   = ("avg_ticket_price", "mean"),
).reset_index()

df = df.merge(network_monthly, on=["year", "month"], how="left")

df["lf_vs_network_avg"]     = df["load_factor"] - df["network_avg_lf"]
df["price_vs_network_avg"]  = df["avg_ticket_price"] - df["network_avg_price"]
df["is_post_covid"]         = (df["year"] >= 2022).astype(int)
df["is_covid_period"]       = ((df["year"] == 2020) |
                                ((df["year"] == 2021) & (df["month"] <= 6))).astype(int)
df["recovery_phase"]        = (
    (df["year"] == 2021) & (df["month"] >= 7) |
    (df["year"] == 2022)
).astype(int)

print(f"[✓] +{7} features réseau ajoutées")

# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 6 — LAG FEATURES & ROLLING STATS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Lag Features & Rolling Statistics...")

df_sorted = df.sort_values(["route", "year", "month"]).copy()

for lag in [1, 2, 3, 6, 12]:
    df_sorted[f"lf_lag_{lag}m"] = df_sorted.groupby("route")["load_factor"].shift(lag)
    df_sorted[f"price_lag_{lag}m"] = df_sorted.groupby("route")["avg_ticket_price"].shift(lag)

for window in [3, 6, 12]:
    df_sorted[f"lf_rolling_mean_{window}m"] = (
        df_sorted.groupby("route")["load_factor"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df_sorted[f"lf_rolling_std_{window}m"] = (
        df_sorted.groupby("route")["load_factor"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
    )

df_sorted["lf_mom_change"]  = df_sorted.groupby("route")["load_factor"].diff(1)
df_sorted["lf_yoy_change"]  = df_sorted.groupby("route")["load_factor"].diff(12)

df = df_sorted.copy()
print(f"[✓] +{16} lag/rolling features ajoutées")

# ══════════════════════════════════════════════════════════════════════════════
# ENCODAGE VARIABLES CATÉGORIELLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[INFO] Encodage variables catégorielles...")

weather_map = {"CLEAR": 0, "CLOUDY": 1, "RAIN": 2, "SNOW": 3}
df["weather_encoded"] = df["weather_condition"].map(weather_map).fillna(0)

from sklearn.preprocessing import LabelEncoder

le_route  = LabelEncoder()
le_origin = LabelEncoder()
le_dest   = LabelEncoder()

df["route_encoded"]  = le_route.fit_transform(df["route"])
df["origin_encoded"] = le_origin.fit_transform(df["origin"])
df["dest_encoded"]   = le_dest.fit_transform(df["dest"])

# Sauvegarder les encodeurs
import pickle
encoders = {
    "route":  le_route,
    "origin": le_origin,
    "dest":   le_dest
}
with open(FEATURES_DIR / "label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print(f"[✓] Encodeurs sauvegardés")

# ══════════════════════════════════════════════════════════════════════════════
# SÉLECTION FEATURES FINALES POUR ML
# ══════════════════════════════════════════════════════════════════════════════
print("\n[INFO] Sélection features finales...")

ML_FEATURES = [
    # Temporelles
    "year", "month", "day_of_week", "quarter",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_monday", "is_friday",
    "is_summer", "is_winter", "is_spring", "is_fall",
    "is_peak_travel", "is_holiday_period",
    "is_thanksgiving", "is_xmas_newyear", "is_july4",

    # Vol
    "distance", "seats", "avg_ticket_price",
    "price_per_mile", "revenue_per_seat", "yield_metric",
    "is_long_haul", "is_medium_haul", "is_short_haul",
    "weather_encoded",

    # Route
    "route_avg_lf", "route_std_lf", "route_max_lf",
    "route_avg_price", "route_avg_distance",
    "route_lf_cv", "route_popularity_rank",
    "route_encoded",

    # Aéroport
    "origin_hub_tier", "dest_hub_tier",
    "hub_to_hub", "hub_to_spoke", "is_atl_flight",
    "origin_avg_lf", "origin_n_routes",
    "origin_encoded", "dest_encoded",

    # Réseau
    "network_avg_lf", "network_avg_price",
    "price_vs_network_avg",
    "is_post_covid", "is_covid_period", "recovery_phase",
    "covid_impact_factor", "seasonality_index",

    # Lag features
    "lf_lag_1m", "lf_lag_2m", "lf_lag_3m",
    "lf_lag_6m", "lf_lag_12m",
    "lf_rolling_mean_3m", "lf_rolling_mean_6m", "lf_rolling_mean_12m",
    "lf_rolling_std_3m", "lf_rolling_std_6m",
    "lf_mom_change", "lf_yoy_change",
]

TARGET = "load_factor"

# Dataset ML final
df_ml = df[ML_FEATURES + [TARGET]].dropna()

print(f"\n[✓] Features sélectionnées : {len(ML_FEATURES)}")
print(f"[✓] Dataset ML final       : {len(df_ml):,} lignes")
print(f"[✓] Target                 : '{TARGET}' (Load Factor %)")
print(f"[✓] Taux de rétention      : {len(df_ml)/len(df)*100:.1f}%")

# ── Sauvegardes ───────────────────────────────────────────────────────────────
df_ml.to_csv(FEATURES_DIR / "delta_features_ml.csv", index=False)
df.to_csv(FEATURES_DIR / "delta_features_full.csv", index=False)

# Feature metadata
feature_meta = {
    "target":           TARGET,
    "n_features":       len(ML_FEATURES),
    "n_samples":        len(df_ml),
    "features":         ML_FEATURES,
    "feature_groups": {
        "temporal":     20,
        "flight":       9,
        "route":        8,
        "airport":      7,
        "network":      7,
        "lag_rolling":  12
    }
}
with open(FEATURES_DIR / "feature_metadata.json", "w") as f:
    json.dump(feature_meta, f, indent=2)

print(f"\n[✓] Fichiers sauvegardés :")
print(f"    → data/features/delta_features_ml.csv")
print(f"    → data/features/delta_features_full.csv")
print(f"    → data/features/feature_metadata.json")
print(f"    → data/features/label_encoders.pkl")

# ── Rapport résumé ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"{'Groupe':<20} {'Features':>10}")
print("-"*32)
for group, count in feature_meta["feature_groups"].items():
    print(f"{'  '+group:<20} {count:>10}")
print("-"*32)
print(f"{'  TOTAL':<20} {sum(feature_meta['feature_groups'].values()):>10}")
print(f"\n[✓] ÉTAPE 4 COMPLÈTE — {len(ML_FEATURES)} features prêtes pour ML")