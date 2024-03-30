import pandas as pd, numpy as np, streamlit as st
from prediction import pred

st.header("Traffic Accident Prediction")
col1, col2 = st.columns(2)
with col1:
    st.text("Environment Details")
    WC = st.text_input(label="Weather conditions")
    LoM = st.text_input(label="Lanes or Medians")
    ToJ = st.text_input(label="Types of Junction")
    ToC = st.text_input(label="Type of collision")
    RST = st.text_input(label="Road surface type")
    LC = st.text_input(label="Light conditions")
    PM = st.text_input(label="Pedestrian movement")
with col2:
    st.text("Vehicle-Driver Details")
    AboD = st.text_input(label="Age band of Driver")
    SoD = st.text_input(label="Sex of Driver") 
    VDR = st.text_input(label="Vehicle Driver Relation")
    DE = st.text_input(label="Driving Experience")
    VM = st.text_input(label="Vehicle movement")

CoA = st.text_input(label="Cause of accident")

new_data = {"Age_band_of_driver" : AboD,
            "Sex_of_driver" : SoD,
            "Vehicle_driver_relation" : VDR, 
            "Driving_experience" : DE, 
            "Lanes_or_Medians" : LoM,
            "Types_of_Junction" : ToJ, 
            "Road_surface_type" : RST, 
            "Light_conditions" : LC,
            "Weather_conditions" : WC, 
            "Type_of_collision" : ToC, 
            "Vehicle_movement" : VM,
            'Pedestrian_movement' : PM, 
            'Cause_of_accident' : CoA}

if st.button("Predict accident severity"):
    List = pd.DataFrame([new_data])
    result, result1, result2 = pred(List)
    if result == 2:
        st.text("Logistic Regression: Slight Injury (Recommended)")
    elif result == 1:
        st.text("Logistic Regression: Serious Injury (Recommended)")
    else:
        st.text("Logistic Regression: Fatal Injury (Recommended)")
        
    if result1 == 2:
        st.text("\nK Nearest Neighbours: Slight Injury")
    elif result1 == 1:
        st.text("\nK Nearest Neighbours: Serious Injury")
    else:
        st.text("\nK Nearest Neighbours: Fatal Injury")
    
    if result2 == 2:
        st.text("\nNaive Bayes: Slight Injury")
    elif result2 == 1:
        st.text("\nNaive Bayes: Serious Injury")
    else:
        st.text("\nNaive Bayes: Fatal Injury")