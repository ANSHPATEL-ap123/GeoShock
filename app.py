import streamlit as st
import joblib
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GeoShock | AI Disaster Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_assets():
    classifier = joblib.load('impact_classifier.pkl')
    regressor = joblib.load('damage_estimator.pkl')
    knn_engine = joblib.load('knn_similarity_engine.pkl')
    history_df = pd.read_csv('historical_reference.csv')
    return classifier, regressor, knn_engine, history_df

try:
    classifier_model, regressor_model, knn_engine, history_df = load_assets()
except Exception as e:
    st.error(f"Error loading models or dataset files: {e}")
    st.stop()

# --- HEADER SECTION ---
st.title("🌍 GeoShock: Intelligent Disaster Analytics")
st.caption("AI-Powered Seismic Threat Classification, Economic Damage Estimation & Historical Pattern Matching")
st.divider()

# --- SIDEBAR INPUT CONTROLS ---
with st.sidebar:
    st.header("⚙️ Seismic Input Parameters")
    st.write("Adjust the parameters below to simulate a hypothetical earthquake.")
    
    magnitude = st.slider(
        "Magnitude (Richter Scale)",
        min_value=4.0,
        max_value=9.5,
        value=6.5,
        step=0.1,
        help="Higher values indicate stronger seismic energy release."
    )
    
    focal_depth = st.slider(
        "Focal Depth (km)",
        min_value=0.0,
        max_value=700.0,
        value=25.0,
        step=1.0,
        help="Shallow earthquakes (<70 km) generally cause significantly more surface destruction."
    )
    
    st.info("💡 Adjusting these sliders instantly recalculates AI model predictions.")

# --- INFERENCE ENGINE ---
# Create input dataframe matching exact model feature names
input_data = pd.DataFrame({
    'magnitude': [magnitude],
    'focal_depth': [focal_depth]
})

# Run predictive models
raw_impact = classifier_model.predict(input_data)[0]
economic_damage = max(0.0, float(regressor_model.predict(input_data)[0]))

# KNN Similarity Search
distances, indices = knn_engine.kneighbors(input_data)
match_index = indices[0][0]
match = history_df.iloc[match_index]

# Map impact tier logic reliably
impact_label = f"Tier {raw_impact}"
if raw_impact == 2 or (raw_impact == 0 and economic_damage > 1000):
    impact_title = "Severe Human Impact"
    impact_level = "Severe"
elif raw_impact == 1 or economic_damage > 250:
    impact_title = "Moderate Human Impact"
    impact_level = "Moderate"
else:
    impact_title = "Minimal Human Impact"
    impact_level = "Minimal"

# --- 1. THREAT ASSESSMENT DASHBOARD ---
st.subheader("1. Threat Assessment Summary")

col1, col2, col3 = st.columns([1.2, 1.2, 1.4])

with col1:
    with st.container(border=True):
        st.markdown("### 💰 Economic Damage")
        st.metric(
            label="Estimated Loss (USD)",
            value=f"${economic_damage:,.2f} M"
        )

with col2:
    with st.container(border=True):
        st.markdown("### 🚨 Impact Severity")
        if impact_level == "Severe":
            st.error(f"**{impact_title}** ({impact_label})")
        elif impact_level == "Moderate":
            st.warning(f"**{impact_title}** ({impact_label})")
        else:
            st.success(f"**{impact_title}** ({impact_label})")

with col3:
    with st.container(border=True):
        st.markdown("### 📐 Active Parameters")
        st.write(f"**Magnitude:** {magnitude} Richter")
        st.write(f"**Focal Depth:** {focal_depth} km")

st.divider()

# --- 2. HISTORICAL PATTERN MATCHING ---
st.subheader("2. Historical Pattern Search (KNN Recommendation Engine)")

with st.container(border=True):
    st.markdown(
        f"📍 **Nearest Historical Match:** "
        f"Statistically similar to the **{int(match['year'])} {match['name']} Earthquake**, "
        f"which caused **${match['damage']:,.2f} Million** in economic damage."
    )

st.divider()

# --- 3. MODEL EXPLAINABILITY & ANALYTICS ---
st.subheader("3. Model Decision Analytics (Feature Importance)")
st.write("Dynamic feature weights extracted from the Random Forest Regressor:")

# Extract feature importances
importances = regressor_model.feature_importances_
analytics_df = pd.DataFrame({
    'Importance Score': importances
}, index=['Magnitude', 'Focal Depth'])

# Render bar chart ONCE
st.bar_chart(analytics_df, use_container_width=True)

st.caption("Architecture: Stacked Ensemble Classification | Random Forest Regression | K-Nearest Neighbors Spatial Search")
