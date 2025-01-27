import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# Function to cache data loading and processing
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

# Load the data and target names
df, target_name = load_data()

# Initialize and train the model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Sidebar input features for prediction
st.sidebar.title("Input features")
sepal_length = st.sidebar.slider(
    "Sepal length", 
    float(df['sepal length (cm)'].min()), 
    float(df['sepal length (cm)'].max())
)
sepal_width = st.sidebar.slider(
    "Sepal width", 
    float(df['sepal width (cm)'].min()), 
    float(df['sepal width (cm)'].max())
)
petal_length = st.sidebar.slider(
    "Petal length", 
    float(df['petal length (cm)'].min()), 
    float(df['petal length (cm)'].max())
)
petal_width = st.sidebar.slider(
    "Petal width", 
    float(df['petal width (cm)'].min()), 
    float(df['petal width (cm)'].max())
)

# Prepare input data for prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make the prediction
prediction = model.predict(input_data)
predicted_species = target_name[prediction[0]]

# Display the result
st.write("Prediction")
st.write(f"The predicted species is: {predicted_species}")

# Image paths for the Iris species
image_dict = {
    "setosa": "images/setosa.jpg",
    "versicolor": "images/versicolour.jpg",
    "virginica": "images/virginica.jpg"
}

# Display the corresponding image
if predicted_species.lower() in image_dict:
    img = Image.open(image_dict[predicted_species.lower()])
    st.image(img, caption=f"Predicted species: {predicted_species}", use_container_width=True)

