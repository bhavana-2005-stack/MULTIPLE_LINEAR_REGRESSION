import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# page config
st.set_page_config(
    page_title="Multiple Linear Regression",
    layout="centered"
)

# load css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# title
st.markdown(
    """
    <div class="card">
        <h1>Multiple Linear Regression</h1>
        <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> and <b>Size</b> using Multiple Linear Regression...</p>
    </div>
    """,
    unsafe_allow_html=True
)

#load data
@st.cache_data

def load_data():
    return sns.load_dataset("tips")
df=load_data()

#Dataset Preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[["total_bill","size","tip"]].head())
st.markdown('</div>',unsafe_allow_html= True)

#Prepare Data
x,y=df[['total_bill','size']],df['tip']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#Train Model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Metrics
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-x.shape[1]-1)

#Visualization
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("total Bill vs tip (with MLR)")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6)
ax.plot(df["total_bill"],model.predict(scaler.transform(x)),color='red')
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)

st.markdown('<div>',unsafe_allow_html=True)

#Performance Metrics
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Model Performance Metrics")
c1,c2=st.columns(2)
c1.metric("mae:",f"{mae:.2f}")
c2.metric("rmse:",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("r2 Score:",f"{r2:.2f}")      
c4.metric("Adj r2 Score:",f"{adj_r2:.2f}")

#m & c
st.markdown(f"""
            <div class="card">
            <h3>Model Interception</h3>
            <p><b>co-efficient(Total bill):</b>{model.coef_[0]:.3f}</p>
            <p><b>co-efficient(Group Size):</b>{model.coef_[1]:.3f}</p>
            <p><b>intercept:</b>{model.intercept_:.3f}</p>
            <p> Tip depends on both Total Bill and Size of the group.</p>
            </div>
""",unsafe_allow_html=True)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Select Total Bill",
    float(df['total_bill'].min()),
    float(df['total_bill'].max()),
    30.0
)

size=st.slider(
    "Select Group Size",
    int(df['size'].min()),
    int(df['size'].max()),
    2
)
input_scaled = scaler.transform([[bill, size]])

tip = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: {tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)