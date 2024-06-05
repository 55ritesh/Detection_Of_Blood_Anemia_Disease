import streamlit as st
from PIL import Image
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model_path = 'random_forest_model.pkl'  # Path to your trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the function to preprocess input data
def preprocess_data(hemoglobin, gender, mcv):
    # Convert gender to numeric value
    gender_mapping = {'Male': 0, 'Female': 1}
    gender = gender_mapping.get(gender, 0)  # Default to 0 if gender is not found

    # Ensure non-negative values for hemoglobin and MCV
    hemoglobin = max(hemoglobin, 0)
    mcv = max(mcv, 0)

    # Create a dataframe with the input data
    data = {'Gender': [gender], 'Hemoglobin': [hemoglobin], 'MCV': [mcv]}
    df = pd.DataFrame(data)

    return df

# Define the function to predict anemia
def predict_anemia(hemoglobin, gender, mcv):
    # Preprocess the input data
    df = preprocess_data(hemoglobin, gender, mcv)

    # Predict anemia using the trained model
    prediction = model.predict(df)

    # Return the prediction
    return prediction[0]

# Load image
img1 = Image.open("Iron-deficiency-anemia-symptoms-caused-by-lack-of-iron-and-treatment.png")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        color: #31333f;
    }
    .stButton > button {
        background-color: #6c63ff;
        color: white;
    }
    .stButton > button:hover {
        background-color: #483d8b;
        color: white;
    }
    .stNumberInput input {
        border-color: #6c63ff;
        color: #31333f;
    }
    .stSelectbox select {
        border-color: #6c63ff;
        color: #31333f;
    }
    .stAlert {
        background-color: #fff3cd;
        color: #856404;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
    }
    .title {
        color: #6c63ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create the Streamlit app
def main():
    # Set the title and description
    st.title("Anemia Detection")
    st.image(img1, width=400)  # Adjust the width to make the image smaller
    st.write("This app helps detect anemia based on input values of hemoglobin, gender, and MCV.")

    # Add a divider
    st.markdown("<hr>", unsafe_allow_html=True)

    # Create a sidebar for input fields
    st.sidebar.header("Enter the required information:")
    hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", value=12.0, min_value=0.0, step=0.1)
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'], index=0)
    mcv = st.sidebar.number_input("MCV (fL)", value=90.0, min_value=0.0, step=0.1)

    # Create a button to detect anemia
    st.subheader("Result")
    # Display the prediction
    if st.sidebar.button("Detect Anemia"):
        try:
            if hemoglobin >= 0 and mcv >= 0:
                # Call the predict_anemia function to get the prediction
                prediction = predict_anemia(hemoglobin, gender, mcv)

                # Map the prediction to the corresponding label
                prediction_label = 'Anemic' if prediction == 1 else 'Non-Anemic'

                # Display the prediction with enhanced styling
                st.markdown(f"<p style='font-size: 18px; color: {'red' if prediction == 1 else 'green'};'><b>The person is likely to be {prediction_label}</b></p>", unsafe_allow_html=True)
            else:
                st.warning("Please enter valid values.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Add a footer
    st.markdown("<hr>", unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
