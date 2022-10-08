import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

pickle_in = open("finalized_model.sav", "rb")
regressor = pickle.load(pickle_in)


# @app.route('/')
def welcome():
    return "Welcome All"


# @app.route('/predict',methods=["Get"])
def predict_strength(Cement, Blast_Furnace_Slag, Fly_Ash, Water, Superplasticizer, Coarse_Aggregate, Fine_Aggregate,Age):
    """Let's predict strength of cement.
    This is using docstrings for specifications.
    ---
    parameters:
      - name: Cement
        in: query
        type: number
        required: true
      - name: Blast_Furnance_Slag
        in: query
        type: number
        required: true
        name: Fly_Ash
        in: query
        type: number
        required: true
      - name: Water
        in: query
        type: number
        required: true
      - name: Superplasticizer
        in: query
        type: number
        required: true
        name: Coarse_Aggregate
        in: query
        type: number
        required: true
        name: Fine_Aggregate
        in: query
        type: number
        required: true
        name: Age
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """

    prediction = regressor.predict(np.array([[Cement, Blast_Furnace_Slag, Fly_Ash, Water, Superplasticizer, Coarse_Aggregate, Fine_Aggregate, Age]]))

    return prediction[0]


def main():
    st.title("Strength Of Cement")
    st.image("image.jpg", width=700)
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Cemenet Strength Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Cement = st.text_input("Cement(In kg in a m^3)")
    Blast_Furnace_Slag = st.text_input("Blast_Furnace_Slag(In kg in a m^3)")
    Fly_Ash = st.text_input("Fly_Ash(In kg in a m^3)")
    Water = st.text_input("Water(In kg in a m^3)")
    Superplasticizer = st.text_input("Superplasticizer(In kg in a m^3)")
    Coarse_Aggregate = st.text_input("Coarse_Aggregate(In kg in a m^3)")
    Fine_Aggregate = st.text_input("Fine_Aggregate(In kg in a m^3)")
    Age = st.text_input("Age(In Day)")

    result = ""
    if st.button("Predict"):
        result = predict_strength(float(Cement), float(Blast_Furnace_Slag), float(Fly_Ash), float(Water),float(Superplasticizer), float(Coarse_Aggregate), float(Fine_Aggregate), float(Age))

    st.success('The output is {}'.format(result))

    if st.button("About ML App"):
        st.text(
            "Regression model to predict the concrete compressive strength based on the different features in the training data")
        st.text("Built with Streamlit")

    if st.button("Input Dictionary"):
        st.text("Cement : Quantity of Cement present in kg in a m3 mixture")
        st.text("Blast_Furnance_Slag :  Quantity of Blast_Furnance_Slag present in kg in a m3 mixture")
        st.text("Fly_Ash :  Quantity of Fly_Ash present in kg in a m3 mixture")
        st.text("Water : Quantity of Fly_Ash present in kg in a m3 mixture")
        st.text("Superplasticizer : Quantity of Superplasticizer in kg in a m3 mixture")
        st.text("Coarse_Aggregate : Quantity of Coarse_Aggregate in kg in a m3 mixture")
        st.text("Fine_Aggregate : Quantity of Fine_Aggregate in kg in a m3 mixture")
        st.text("Age : Age in days")
        st.text("Concrete compressive strength: Output Variable - Concrete compressive strength in (MPa, megapascals)")

    if st.button("About Author"):
        st.text("Name : Rohit Murkute")
        st.text("Group Leader")
        st.text("Email : mr324091@gmail.com")
        st.text("Oragnization : Data Science Student at SPPU")

        st.text("Name : Yash Bhambere")
        st.text("Group Member")
        st.text("Email : yashbhammbere99@gmail.com")
        st.text("Oragnization : Data Science Student at SPPU")

        st.text("Name : Akshay Deshmukh")
        st.text("Group Member")
        st.text("Email : abdeshmukh232323@gmail.com")
        st.text("Oragnization : Data Science Student at SPPU")


if __name__ == '__main__':
    main()
