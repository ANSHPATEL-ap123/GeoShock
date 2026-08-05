# 🌍 GeoShock: Intelligent Disaster Analytics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

**GeoShock** is an end-to-end, AI-powered seismic analytics platform designed to predict human impact, estimate economic damage, and identify historical disaster patterns using real-time earthquake parameters. 

Built with a focus on high-performance algorithmic predictions and interactive data visualization, this application serves as a comprehensive threat assessment dashboard for hypothetical seismic events.

---

## 🚀 Live Application

**[🔗 Access the Live GeoShock Dashboard Here](https://geoshock.streamlit.app/)**

*(Optional: Replace the placeholder below with a screenshot of your dark-mode UI)*
> 📸 *Insert UI Screenshot Here: `![GeoShock UI](link-to-image)`*

---

## 🧠 Core Architecture

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Responsive, dark-mode native threat assessment dashboard. |
| **Impact Classification** | Stacked Ensemble | Categorizes human impact severity (Minimal, Moderate, Severe). |
| **Damage Regression** | Random Forest | Estimates total economic destruction in USD (Millions) handling heavy outliers. |
| **Recommendation Engine** | K-Nearest Neighbors (KNN) | Spatial search to find the closest historical earthquake match based on Euclidean distance. |
| **Data Processing** | Pandas, Scikit-Learn | Feature scaling, label encoding, and automated pipeline structuring. |

---

## 🔍 Machine Learning Pipeline

GeoShock processes inputs (Magnitude and Focal Depth) through three distinct machine learning models simultaneously:

1. **The Human Impact Classifier:** 
   A Stacked Ensemble model trained on historical seismic data. It understands the non-linear physics of earthquakes, accurately predicting that deep-focus earthquakes (e.g., >100km) cause less surface destruction than shallow-focus events, regardless of magnitude.
2. **The Economic Damage Estimator:** 
   A Random Forest Regressor optimized to handle extreme statistical outliers, predicting financial damage in the millions.
3. **The Historical Similarity Engine:** 
   An instance-based learning algorithm utilizing $K=1$ Euclidean distance calculations to map the user's hypothetical input against a vector space of historical disasters, returning the closest statistical match.

---

