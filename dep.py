import streamlit as st
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split

df=sns.load_dataset("iris")
target=df["species"]
inputs=df[df.columns[:-1]]

x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=.2,random_state=30)
lr=LogisticRegression()
lr.fit(x_train,y_train)

feature_names=['sepal length', 'sepal width', 'petal length', 'petal width']
image_urls = {
    "setosa": "https://imgs.search.brave.com/Jl2QIG7R9bLzVK2PstflfDiGANObxDAmZX2NyXygr3U/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy84/Lzg2L0lyaXNfc2V0/b3NhLkpQRw",
    "versicolor": "https://imgs.search.brave.com/GCL7wzIUE4IDRRJhLnqrCPtNXNWcu87wrRyjOGWnkFI/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9pLmV0/c3lzdGF0aWMuY29t/LzMzNTQxNTY4L2Mv/NDI0LzQyNC8wLzM3/L2lsLzdjMDI3NS8z/ODExNTI5MTc5L2ls/XzYwMHg2MDAuMzgx/MTUyOTE3OV9iaGxz/LmpwZw",
    "virginica": "https://imgs.search.brave.com/Yk0mm_5gTeEt2vbU5sf6hX3y85bta6vrh933CQr0O6w/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9zd2Jp/b2RpdmVyc2l0eS5v/cmcvaW1nbGliL3Nl/aW5ldC9taWR3ZXN0/L0lyaWRhY2VhZS8y/MDIyMTAvSXJpc192/aXJnaW5pY2FfMV81/LTIyLTIwMDVfVVNf/XzE2NjUyODE0Njdf/dG4uanBn",
    "unknown": "https://placehold.co/400x300/EEE/31343C?text=Unknown+Flower&font=Montserrat",
}


# --- Sidebar Elements ---
st.sidebar.header("Flower Characteristics")
sepal_length = st.sidebar.slider(feature_names[0], 4.0, 8.0, 5.8, 0.1)
sepal_width = st.sidebar.slider(feature_names[1], 2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider(feature_names[2], 1.0, 7.0, 3.8, 0.1)
petal_width = st.sidebar.slider(feature_names[3], 0.1, 2.5, 1.2, 0.1)

# --- Prediction and Image Display ---
st.title("Iris Flower Prediction App")

# Prepare input for the model
input_data = [sepal_length, sepal_width, petal_length, petal_width]



# Make prediction
prediction = lr.predict([input_data])
image_url = image_urls.get(prediction[0], image_urls["unknown"])

st.write(f"Predicted Flower Type: {prediction[0]}")
st.image(image_url, caption=f"{prediction[0].capitalize()} Flower", width=400)

st.subheader("Input Features:")
st.write(f"{feature_names[0]}: {sepal_length} cm")
st.write(f"{feature_names[1]}: {sepal_width} cm")
st.write(f"{feature_names[2]}: {petal_length} cm")
st.write(f"{feature_names[3]}: {petal_width} cm")
