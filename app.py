import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

plt.style.use("seaborn-v0_8")

# ------------------------------
# 1. Load data, model, and build monthly historical + forecast
# ------------------------------
@st.cache_resource
def load_data_and_forecast():
    # Load cleaned Pasig dataset
    df_model = pd.read_csv("data/pasig_clean_for_model.csv")
    df_model["datetime"] = pd.to_datetime(df_model["datetime"], errors="coerce")

    # Load model + feature names
    rf_model = joblib.load("models/rf_pasig_apparent_temp.pkl")
    with open("models/feature_list.json", "r") as f:
        feature_names = json.load(f)

    # Ensure month column exists for grouping
    df_model["month"] = df_model["datetime"].dt.month
    df_model["year"] = df_model["datetime"].dt.year

    # ---------- Historical monthly averages ----------
    df_monthly_hist = (
        df_model.groupby(["year", "month"])["apparent_temperature_max"]
        .mean()
        .reset_index()
    )
    df_monthly_hist["date"] = pd.to_datetime(
        df_monthly_hist["year"].astype(str) + "-" +
        df_monthly_hist["month"].astype(str) + "-01"
    )
    df_monthly_hist["source"] = "Historical"
    df_monthly_hist.rename(
        columns={"apparent_temperature_max": "value"},
        inplace=True
    )

    # ---------- Prepare monthly feature means (template per month) ----------
    monthly_feature_means = df_model.groupby("month")[feature_names].mean()

    # ---------- Forecast Apr–Dec 2025 ----------
    future_dates_2025 = pd.date_range(start="2025-04-01", end="2025-12-31", freq="D")
    rows_2025 = []
    for dt in future_dates_2025:
        m = dt.month
        avg_row = monthly_feature_means.loc[m].copy()
        avg_row["month"] = m
        rows_2025.append(avg_row)

    X_future_2025 = pd.DataFrame(rows_2025, index=future_dates_2025)
    y_future_2025 = rf_model.predict(X_future_2025[feature_names])

    df_forecast_2025 = pd.DataFrame({
        "datetime": future_dates_2025,
        "predicted_apparent_temperature_max": y_future_2025
    })

    # ---------- Forecast entire 2026 ----------
    future_dates_2026 = pd.date_range(start="2026-01-01", end="2026-12-31", freq="D")
    rows_2026 = []
    for dt in future_dates_2026:
        m = dt.month
        avg_row = monthly_feature_means.loc[m].copy()
        avg_row["month"] = m
        rows_2026.append(avg_row)

    X_future_2026 = pd.DataFrame(rows_2026, index=future_dates_2026)
    y_future_2026 = rf_model.predict(X_future_2026[feature_names])

    df_forecast_2026 = pd.DataFrame({
        "datetime": future_dates_2026,
        "predicted_apparent_temperature_max": y_future_2026
    })

    # ---------- Combine forecast daily ----------
    df_forecast = pd.concat([df_forecast_2025, df_forecast_2026], ignore_index=True)
    df_forecast["year"] = df_forecast["datetime"].dt.year
    df_forecast["month"] = df_forecast["datetime"].dt.month

    df_monthly_forecast = (
        df_forecast.groupby(["year", "month"])["predicted_apparent_temperature_max"]
        .mean()
        .reset_index()
    )
    df_monthly_forecast["date"] = pd.to_datetime(
        df_monthly_forecast["year"].astype(str) + "-" +
        df_monthly_forecast["month"].astype(str) + "-01"
    )
    df_monthly_forecast["source"] = "Forecast"
    df_monthly_forecast.rename(
        columns={"predicted_apparent_temperature_max": "value"},
        inplace=True
    )

    # ---------- Combine historical + forecast ----------
    df_all = pd.concat(
        [df_monthly_hist, df_monthly_forecast],
        ignore_index=True
    )

    return df_all


# ------------------------------
# 2. Streamlit UI
# ------------------------------
st.title("🌤️ Pasig City Apparent Temperature – Historical & Forecast")
st.write(
    "This dashboard shows **monthly average apparent temperature** in Pasig City "
    "from **2020 to 2026**, including forecasts for **Apr–Dec 2025** and **all of 2026** "
    "based on a Random Forest regression model."
)

df_all = load_data_and_forecast()

years_available = sorted(df_all["year"].unique())
selected_year = st.selectbox("Select Year", years_available, index=years_available.index(2025))

st.markdown(
    f"Showing **monthly averages for {selected_year}**. "
    "Blue = historical, Orange = forecasted months."
)

# Filter by selected year
df_year = df_all[df_all["year"] == selected_year].sort_values("date")

# ------------------------------
# 3. Plot with seasons + historical vs forecast
# ------------------------------
fig, ax = plt.subplots(figsize=(12,5))

# Historical months for this year
mask_hist = df_year["source"] == "Historical"
mask_fore = df_year["source"] == "Forecast"

if mask_hist.any():
    ax.plot(
        df_year.loc[mask_hist, "date"],
        df_year.loc[mask_hist, "value"],
        label="Historical Monthly Avg",
        color="steelblue",
        marker="o"
    )

if mask_fore.any():
    ax.plot(
        df_year.loc[mask_fore, "date"],
        df_year.loc[mask_fore, "value"],
        label="Forecast Monthly Avg",
        color="orange",
        marker="o",
        linewidth=3
    )

# ---------- Philippine seasons for this year ----------
year = int(selected_year) # type: ignore

# Cool Dry: Dec, Jan, Feb
cool_spans = [
    (datetime(year, 1, 1), datetime(year, 2, 28)),
    (datetime(year, 12, 1), datetime(year, 12, 31)),
]

# Hot Dry: Mar–May
hot_span = (datetime(year, 3, 1), datetime(year, 5, 31))

# Rainy: Jun–Nov
rain_span = (datetime(year, 6, 1), datetime(year, 11, 30))

for start, end in cool_spans:
    ax.axvspan(start, end, color="lightskyblue", alpha=0.2, label="Cool Dry (Dec–Feb)") # type: ignore

ax.axvspan(
    hot_span[0], hot_span[1], # type: ignore
    color="navajowhite", alpha=0.25, label="Hot Dry (Mar–May)"
)

ax.axvspan(
    rain_span[0], rain_span[1], # type: ignore
    color="lightgreen", alpha=0.2, label="Rainy (Jun–Nov)"
)

# Avoid duplicated legend labels for seasons
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), fontsize=9)

# Formatting
ax.set_title(f"Monthly Apparent Temperature – Pasig City ({year})", fontsize=13)
ax.set_xlabel("Month")
ax.set_ylabel("Apparent Temperature Max (°C)")
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.xticks(rotation=0)

plt.tight_layout()
st.pyplot(fig)

# Info note
st.info(
    "🔵 Historical data is used for 2020–2024 and Jan–Mar 2025.\n"
    "🟠 Forecasts are shown for Apr–Dec 2025 and all months of 2026.\n"
    "Shaded regions indicate Philippine climate seasons: Cool Dry (Dec–Feb), "
    "Hot Dry (Mar–May), and Rainy Season (Jun–Nov)."
)
