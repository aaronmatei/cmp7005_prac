"""
Beijing Air Quality — Streamlit Application
============================================
CMP7005 Programming for Data Analysis — PRAC1 (Task 4)

Multi-page GUI built with Streamlit. Pages:
    1. Home              — project context and dataset summary
    2. Dataset           — explore the merged dataset (filter, search, describe)
    3. Visualisations    — interactive univariate / bivariate / multivariate charts
    4. Prediction        — predict PM2.5 with any of the 4 trained models,
                           in either Manual mode or Sample-from-data mode

Run:
    streamlit run app/app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------------------------------------------------------
# Page config + light styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Beijing Air Quality — CMP7005",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)
sns.set_style("whitegrid")

# -----------------------------------------------------------------------------
# Constants — kept at the top so they're easy to change
# -----------------------------------------------------------------------------
DATA_PATH       = "/content/drive/MyDrive/Assessment_Data/processed.csv"
MODEL_DIR       = "/content/drive/MyDrive/Assessment_Data/models"
NUMERIC_FEATURES = ['SO2', 'NO2', 'CO', 'O3',
                    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
                    'hour', 'month', 'day_of_week']
CATEGORICAL_FEATURES = ['wd', 'season', 'station']
AQI_ORDER = ['Good', 'Moderate', 'Unhealthy for Sensitive',
             'Unhealthy', 'Very Unhealthy', 'Hazardous']
AQI_COLOURS = {
    'Good'                   : '#2ecc71',
    'Moderate'               : '#f1c40f',
    'Unhealthy for Sensitive': '#e67e22',
    'Unhealthy'              : '#e74c3c',
    'Very Unhealthy'         : '#9b59b6',
    'Hazardous'              : '#7d3c98',
}
MODEL_FILES = {
    'Random Forest (winner)': 'random_forest.joblib',
    'Decision Tree'         : 'decision_tree.joblib',
    'Linear Regression'     : 'linear_regression.joblib',
    'KNN'                   : 'knn.joblib',
}


# -----------------------------------------------------------------------------
# Cached loaders — Streamlit reruns the script on every interaction, so we
# cache anything expensive (CSV read, joblib load) to keep the UI snappy.
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.error(f"Cannot find {DATA_PATH}. Run the analysis notebook first to generate it.")
        st.stop()
    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    df['aqi_category'] = pd.Categorical(df['aqi_category'],
                                        categories=AQI_ORDER, ordered=True)
    return df


@st.cache_resource(show_spinner="Loading models…")
def load_artefacts():
    """Load all 4 models, the scaler, and the trained feature column order."""
    if not os.path.exists(MODEL_DIR):
        st.error(f"Cannot find {MODEL_DIR}/. Run the analysis notebook first to train the models.")
        st.stop()
    models = {name: joblib.load(os.path.join(MODEL_DIR, fn))
              for name, fn in MODEL_FILES.items()}
    scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.joblib'))
    return models, scaler, columns


def pm25_to_category(v: float) -> str:
    """Apply China MEE PM2.5 24-hour breakpoints (same as the notebook)."""
    if v <= 35:  return 'Good'
    if v <= 75:  return 'Moderate'
    if v <= 115: return 'Unhealthy for Sensitive'
    if v <= 150: return 'Unhealthy'
    if v <= 250: return 'Very Unhealthy'
    return 'Hazardous'


# =============================================================================
# PAGE 1 — HOME
# =============================================================================
def page_home(df: pd.DataFrame):
    st.title("🌫️ Beijing Air Quality — Analysis & Prediction")
    st.markdown(
        "An interactive companion to the CMP7005 PRAC1 notebook. "
        "Use the sidebar to navigate between **Dataset**, **Visualisations**, "
        "and **Prediction** sections."
    )

    st.subheader("Project context")
    st.markdown(
        "- **Source:** PRSA hourly air quality data, 12 nationally controlled "
        "monitoring stations in Beijing (1 March 2013 – 28 February 2017).\n"
        "- **Stations selected:** 2 urban (**Dongsi**, **Wanshouxigong**) "
        "and 2 suburban (**Dingling**, **Huairou**) — chosen following the "
        "categorisation discussed by Xu & Zhang (2020) and Yao et al. (2015).\n"
        "- **Goal:** predict hourly PM2.5 from co-pollutants and meteorology, "
        "and let the user explore the dataset visually."
    )

    st.subheader("Dataset at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",          f"{len(df):,}")
    c2.metric("Columns",       f"{df.shape[1]}")
    c3.metric("Stations",      f"{df['station'].nunique()}")
    c4.metric("Date range",    f"{df['datetime'].min():%Y-%m} → {df['datetime'].max():%Y-%m}")

    st.subheader("Hours per AQI category")
    counts = df['aqi_category'].value_counts().reindex(AQI_ORDER)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(counts.index, counts.values,
                  color=[AQI_COLOURS[c] for c in counts.index])
    ax.set_ylabel("Number of hours")
    ax.set_xticklabels(counts.index, rotation=20, ha='right')
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{int(v):,}", ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)


# =============================================================================
# PAGE 2 — DATASET
# =============================================================================
def page_dataset(df: pd.DataFrame):
    st.title("📊 Dataset Explorer")

    with st.sidebar:
        st.header("Filter")
        stations = st.multiselect("Stations", sorted(df['station'].unique()),
                                  default=sorted(df['station'].unique()))
        years = st.multiselect("Years", sorted(df['year'].unique()),
                               default=sorted(df['year'].unique()))
        seasons = st.multiselect("Seasons",
                                 ['Spring', 'Summer', 'Autumn', 'Winter'],
                                 default=['Spring', 'Summer', 'Autumn', 'Winter'])

    mask = (df['station'].isin(stations)
            & df['year'].isin(years)
            & df['season'].isin(seasons))
    sub = df[mask]

    st.markdown(f"**Filtered rows:** {len(sub):,} of {len(df):,}")

    tab1, tab2, tab3 = st.tabs(["📋 Sample", "📈 Statistics", "🩹 Missing values"])

    with tab1:
        st.dataframe(sub.head(200), use_container_width=True)

    with tab2:
        st.markdown("Numeric summary:")
        st.dataframe(sub.describe().T, use_container_width=True)
        st.markdown("Categorical counts:")
        for col in ['station', 'season', 'wd', 'aqi_category']:
            st.write(f"**{col}**")
            st.write(sub[col].value_counts().head(20))

    with tab3:
        st.markdown("Missing values per column (after preprocessing — should be 0):")
        miss = sub.isnull().sum().sort_values(ascending=False)
        st.dataframe(pd.DataFrame({'missing': miss,
                                   'pct': (miss / max(len(sub), 1) * 100).round(2)}),
                     use_container_width=True)


# =============================================================================
# PAGE 3 — VISUALISATIONS
# =============================================================================
def page_visualisations(df: pd.DataFrame):
    st.title("📈 Visualisations")

    chart = st.sidebar.radio(
        "Chart type",
        ["Distribution (univariate)",
         "Scatter (bivariate)",
         "Boxplot by group",
         "Time-series",
         "Correlation heatmap"],
    )
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    met_vars   = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    all_num    = pollutants + met_vars

    # -------------------------------------------------------------------------
    if chart == "Distribution (univariate)":
        col = st.sidebar.selectbox("Variable", all_num, index=0)
        bins = st.sidebar.slider("Bins", 20, 100, 60)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df[col].dropna(), bins=bins, kde=True,
                     ax=ax, color='steelblue')
        ax.set_title(f"{col} distribution (all stations)")
        st.pyplot(fig)
        st.caption(f"Mean = {df[col].mean():.2f}  |  Median = {df[col].median():.2f}  "
                   f"|  Std = {df[col].std():.2f}")

    # -------------------------------------------------------------------------
    elif chart == "Scatter (bivariate)":
        c1, c2 = st.sidebar.columns(2)
        x = c1.selectbox("X", all_num, index=all_num.index('TEMP'))
        y = c2.selectbox("Y", all_num, index=all_num.index('PM2.5'))
        hue = st.sidebar.selectbox("Colour by", ['None', 'season', 'area_type', 'station'])
        n = st.sidebar.slider("Sample size", 1000, 30000, 10000, step=1000)
        sample = df.sample(min(n, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=sample, x=x, y=y,
                        hue=None if hue == 'None' else hue,
                        alpha=0.4, s=12, ax=ax)
        ax.set_title(f"{y} vs {x}")
        st.pyplot(fig)
        corr = df[[x, y]].corr().iloc[0, 1]
        st.caption(f"Pearson correlation r = {corr:.3f}")

    # -------------------------------------------------------------------------
    elif chart == "Boxplot by group":
        target = st.sidebar.selectbox("Pollutant", pollutants, index=0)
        group  = st.sidebar.selectbox("Group by",
                                      ['season', 'station', 'area_type', 'aqi_category'])
        order = None
        if group == 'season':
            order = ['Spring', 'Summer', 'Autumn', 'Winter']
        elif group == 'aqi_category':
            order = AQI_ORDER
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x=group, y=target, order=order,
                    showfliers=False, palette='Set2', ax=ax)
        ax.set_title(f"{target} by {group}")
        ax.tick_params(axis='x', rotation=20)
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    elif chart == "Time-series":
        target = st.sidebar.selectbox("Variable", all_num, index=0)
        freq   = st.sidebar.radio("Aggregate to", ['Daily', 'Monthly'], index=1)
        rule   = 'D' if freq == 'Daily' else 'ME'
        ts = (df.set_index('datetime')
                .groupby('station')[target]
                .resample(rule).mean()
                .reset_index())
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=ts, x='datetime', y=target, hue='station',
                     ax=ax, marker='o' if freq == 'Monthly' else None)
        ax.set_title(f"{freq} mean {target} by station")
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    elif chart == "Correlation heatmap":
        cols = st.sidebar.multiselect("Columns", all_num, default=all_num)
        if len(cols) < 2:
            st.warning("Pick at least 2 columns.")
        else:
            corr = df[cols].corr()
            fig, ax = plt.subplots(figsize=(min(1 + len(cols), 12),
                                            min(1 + len(cols), 10)))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                        vmin=-1, vmax=1, square=True, ax=ax)
            ax.set_title("Correlation matrix")
            st.pyplot(fig)


# =============================================================================
# PAGE 4 — PREDICTION
# =============================================================================
def build_feature_row(values: dict, feature_columns: list[str],
                      df_for_categories: pd.DataFrame) -> pd.DataFrame:
    """Turn a single dict of input values into a 1-row dataframe with the
    exact column order the model expects, including all one-hot dummies.

    Important: pd.get_dummies on a single row only generates dummies for the
    one category present, not all possible categories. We fix this by casting
    each categorical column to a pandas Categorical with the full known set
    of categories (taken from the training dataframe) before encoding.
    """
    raw = pd.DataFrame([values])
    for c in CATEGORICAL_FEATURES:
        cats = sorted(df_for_categories[c].dropna().unique().tolist())
        raw[c] = pd.Categorical(raw[c], categories=cats)

    encoded = pd.get_dummies(raw, columns=CATEGORICAL_FEATURES, drop_first=True)

    # Defensive: still align to the training column order in case the
    # dropped reference category differs by row (it shouldn't, but be safe).
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[feature_columns]
    return encoded


def page_prediction(df: pd.DataFrame, models, scaler, feature_columns):
    st.title("🔮 PM2.5 Prediction")
    st.caption("Predict the hourly PM2.5 concentration (µg/m³) from co-pollutants "
               "and meteorological conditions.")

    # -- Model selector
    model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()), index=0)
    bundle = models[model_name]
    model, needs_scaling = bundle['model'], bundle['needs_scaling']
    st.sidebar.caption(f"Inputs scaled: **{needs_scaling}**")

    # -- Input mode
    mode = st.sidebar.radio("Input mode",
                            ["Manual entry", "Pick a sample row, then tweak"])

    # ---------------------------------------------------------------------
    # Build the input dictionary based on mode
    # ---------------------------------------------------------------------
    if mode == "Pick a sample row, then tweak":
        idx = st.sidebar.number_input("Row index in dataset",
                                      min_value=0, max_value=len(df)-1,
                                      value=0, step=1)
        seed = df.iloc[int(idx)]
        st.info(f"Seeded from row {int(idx)}  |  Actual PM2.5 = "
                f"**{seed['PM2.5']:.1f} µg/m³**  |  Station = {seed['station']}  "
                f"|  {seed['datetime']}")
        defaults = {c: float(seed[c]) for c in NUMERIC_FEATURES if c in df.columns}
        defaults['wd']      = seed['wd']
        defaults['season']  = seed['season']
        defaults['station'] = seed['station']
    else:
        seed = None
        defaults = {
            'SO2': 10.0, 'NO2': 50.0, 'CO': 1000.0, 'O3': 60.0,
            'TEMP': 15.0, 'PRES': 1015.0, 'DEWP': 5.0, 'RAIN': 0.0, 'WSPM': 1.5,
            'hour': 12, 'month': 6, 'day_of_week': 2,
            'wd': 'NE', 'season': 'Summer', 'station': 'Dongsi',
        }

    st.subheader("Input values")
    # Layout: 4 columns of inputs to keep things compact
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Pollutants**")
        so2 = st.number_input("SO2",  0.0, 500.0, defaults['SO2'])
        no2 = st.number_input("NO2",  0.0, 500.0, defaults['NO2'])
        co  = st.number_input("CO",   0.0, 10000.0, defaults['CO'])
        o3  = st.number_input("O3",   0.0, 500.0, defaults['O3'])

    with col2:
        st.markdown("**Meteorology**")
        temp = st.number_input("TEMP (°C)",  -30.0, 45.0, defaults['TEMP'])
        pres = st.number_input("PRES (hPa)", 980.0, 1050.0, defaults['PRES'])
        dewp = st.number_input("DEWP (°C)",  -40.0, 35.0, defaults['DEWP'])
        rain = st.number_input("RAIN (mm)",  0.0, 100.0, defaults['RAIN'])
        wspm = st.number_input("WSPM (m/s)", 0.0, 20.0, defaults['WSPM'])

    with col3:
        st.markdown("**Time**")
        hour = st.slider("Hour", 0, 23, int(defaults['hour']))
        month = st.slider("Month", 1, 12, int(defaults['month']))
        dow = st.slider("Day of week (0=Mon)", 0, 6, int(defaults['day_of_week']))
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        season = st.selectbox("Season", seasons,
                              index=seasons.index(defaults['season']))

    with col4:
        st.markdown("**Categorical**")
        wd_opts = sorted(df['wd'].dropna().unique().tolist())
        wd = st.selectbox("Wind direction", wd_opts,
                          index=wd_opts.index(defaults['wd']) if defaults['wd'] in wd_opts else 0)
        st_opts = sorted(df['station'].unique().tolist())
        station = st.selectbox("Station", st_opts,
                               index=st_opts.index(defaults['station']))

    values = {
        'SO2': so2, 'NO2': no2, 'CO': co, 'O3': o3,
        'TEMP': temp, 'PRES': pres, 'DEWP': dewp, 'RAIN': rain, 'WSPM': wspm,
        'hour': hour, 'month': month, 'day_of_week': dow,
        'wd': wd, 'season': season, 'station': station,
    }

    st.divider()

    # ---------------------------------------------------------------------
    # Predict
    # ---------------------------------------------------------------------
    if st.button("Predict PM2.5", type="primary", use_container_width=True):
        X_one = build_feature_row(values, feature_columns, df)
        X_for_model = scaler.transform(X_one) if needs_scaling else X_one
        pred = float(model.predict(X_for_model)[0])
        cat = pm25_to_category(pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted PM2.5", f"{pred:.1f} µg/m³")
        c2.metric("AQI category", cat)
        if seed is not None:
            err = pred - seed['PM2.5']
            c3.metric("Error vs actual", f"{err:+.1f}", delta=f"{err:+.1f}")

        # Coloured banner for the AQI category
        st.markdown(
            f"<div style='background:{AQI_COLOURS[cat]};padding:12px;"
            f"border-radius:8px;color:white;font-weight:600;text-align:center'>"
            f"AQI: {cat}</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Show all model predictions for the same input"):
            rows = []
            for name, b in models.items():
                Xm = scaler.transform(X_one) if b['needs_scaling'] else X_one
                rows.append({'Model': name,
                             'Predicted PM2.5': round(float(b['model'].predict(Xm)[0]), 2)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Always show the model leaderboard at the bottom
    with st.expander("Model performance on the held-out test set"):
        comp_path = os.path.join(MODEL_DIR, 'model_comparison.csv')
        if os.path.exists(comp_path):
            st.dataframe(pd.read_csv(comp_path), use_container_width=True)
        else:
            st.caption("model_comparison.csv not found — run the notebook to generate it.")


# =============================================================================
# MAIN — sidebar navigation
# =============================================================================
def main():
    df = load_data()
    models, scaler, feature_columns = load_artefacts()

    st.sidebar.title("🌫️ Beijing Air Quality")
    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Dataset", "Visualisations", "Prediction"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("CMP7005 PRAC1 — Task 4 GUI")

    if page == "Home":
        page_home(df)
    elif page == "Dataset":
        page_dataset(df)
    elif page == "Visualisations":
        page_visualisations(df)
    elif page == "Prediction":
        page_prediction(df, models, scaler, feature_columns)


if __name__ == "__main__":
    main()
