import streamlit as st # type: ignore 
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
import pandas as pd # type: ignore

#Page configuration
st.set_page_config("Simple Linear Regression ", layout="centered")

#Load css 
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")
st.markdown("""
    <div class="card">
        <h1>Simple Linear Regression</h1>
        <p>Predict <b> Tip Amount </b> from <b> Total Bill </b> using Simple Linear Regression...</p>
    </div>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    )

df = load_data()

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

#prepare data
X = df[['total_bill']]  # feature
y = df['tip']           # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

# visualize results
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")
fig,ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)
ax.plot(df["total_bill"], model.predict(scaler.transform(df[["total_bill"]])), color='red')
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig) 
st.markdown('</div>', unsafe_allow_html=True)

# Display evaluation metrics
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Evaluation Metrics")
c1, c2 = st.columns(2)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.2f}")
c4.metric("Adjusted R² Score", f"{adjusted_r2:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# m & c
st.markdown(f""" 
    <div class="card>
            <h3>Model Inception</h3>
            <p><b>Coefficient (m):</b> {model.coef_[0]:.4f}<br>
            <b>Intercept: </b> {model.intercept_:.4f}</p>
            </div>""", unsafe_allow_html=True)
#prediction 
st.markdown('<div class="card">', unsafe_allow_html=True)
bill=st.slider("Total Bill Amount ($)", float(df.total_bill.min()),float(df.total_bill.max()),30.0)
tip=model.predict(scaler.transform([[bill]]))[0]
st.markdown(f"<div class='prediction-box'> Predicted Tip ${tip:.2f}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
