import streamlit as st
import joblib
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GeoShock Analytics", page_icon="🌍", layout="centered")

# --- LOAD MODELS & DATA ---
@st.cache_resource # This keeps the app fast by loading models only once
def load_assets():
    classifier = joblib.load('impact_classifier.pkl')
    regressor = joblib.load('damage_estimator.pkl')
    knn_engine = joblib.load('knn_similarity_engine.pkl')
    history_df = pd.read_csv('historical_reference.csv')
    return classifier, regressor, knn_engine, history_df

classifier_model, regressor_model, knn_engine, history_df = load_assets()

# --- HEADER ---
st.title("🌍 GeoShock: Disaster Analytics")
st.markdown("Predict human impact, estimate economic damage, and find historical similarities using real-time seismic data.")
st.divider()

# --- USER INPUTS ---
st.subheader("1. Enter Seismic Parameters")
magnitude = st.slider("Magnitude (Richter Scale)", min_value=5.0, max_value=9.5, value=6.5, step=0.1)
focal_depth = st.slider("Focal Depth (km)", min_value=0.0, max_value=700.0, value=35.0, step=1.0)

# --- PREDICTION ENGINE ---
if st.button("Run AI Analysis", type="primary"):
    
    # Format input
    input_data = pd.DataFrame({'magnitude': [magnitude], 'focal_depth': [focal_depth]})
    
    # 1. Predictions
    impact_tier = classifier_model.predict(input_data)[0]
    economic_damage = regressor_model.predict(input_data)[0]
    
    # 2. KNN Similarity Match
    distances, indices = knn_engine.kneighbors(input_data)
    match_index = indices[0][0]
    match = history_df.iloc[match_index]
    
    # --- DISPLAY RESULTS ---
    st.divider()
    st.subheader("2. Threat Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Estimated Economic Damage", value=f"${economic_damage:,.2f} M")
        
    with col2:
        if impact_tier == 0:
            st.success("Human Impact: Minimal (Tier 0)")
        elif impact_tier == 1:
            st.warning("Human Impact: Moderate (Tier 1)")
        else:
            st.error("Human Impact: Severe (Tier 2)")
            
    # --- HISTORICAL CONTEXT ---
    st.info(f"**Historical Match:** Statistically similar to the {int(match['year'])} {match['name']} Earthquake, which caused ${match['damage']:,.2f} Million in damage.")
    
    # --- DYNAMIC ANALYTICS ---
    st.divider()
    st.subheader("3. Decision Analytics (Feature Importance)")
    
    importances = regressor_model.feature_importances_
    analytics_df = pd.DataFrame({
        'Importance Score': importances
    }, index=['Magnitude', 'Focal Depth'])
    
    st.bar_chart(analytics_df)
    st.caption("Powered by Stacked Ensemble Classification, Random Forest Regression & KNN Search")