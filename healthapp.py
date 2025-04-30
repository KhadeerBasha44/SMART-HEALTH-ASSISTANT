import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Set custom styles using HTML and CSS
st.markdown("""
    <style>
    .main {
        background-color: black;
    }
    .stSidebar {
        background-color: #c4ddff;
    }
    h1, h2, h3 {
        color: #ff4b4b;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
    }
    .stButton > button:hover {
        background-color: #e83a3a;
    }
    .stTextInput > div > div > input {
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stMarkdown {
        color: #333;
    }
    .stError {
        color: #ff4b4b;
    }
    .stSuccess {
        color: #28a745;
    }
    /* Add this new style */
    .stTooltipIcon {
        color: #ff4b4b !important;
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        diabetes_model = pickle.load(open('best_diabetes_model.sav', 'rb'))
        heart_model = pickle.load(open('best_heart_model.sav', 'rb'))
        parkinsons_model = pickle.load(open('best_parkinsons_model.sav', 'rb'))
        return diabetes_model, heart_model, parkinsons_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

diabetes_model, heart_model, parkinsons_model = load_models()

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                         ['Diabetes Prediction', 
                          'Heart Disease Prediction', 
                          'Parkinsons Prediction'],
                         menu_icon='hospital-fill',
                         icons=['activity', 'heart', 'person'],
                         default_index=0)

# ====================== DIABETES PREDICTION (UPDATED) ======================
if selected == 'Diabetes Prediction':
    st.markdown("<h1>Diabetes Prediction using ML</h1>", unsafe_allow_html=True)
    
    # Getting the input data from the user (3 key features)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=100)
    
    with col2:
        bmi = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    with col3:
        age = st.number_input('Age (years)', min_value=1, max_value=120, value=30)
    
    # Prediction logic - updated to match your second code
    if st.button('Diabetes Test Result'):
        try:
            # Prepare input data
            input_data = pd.DataFrame([[glucose, bmi, age]], 
                                    columns=['Glucose', 'BMI', 'Age'])
            
            # Get prediction and probability
            proba = diabetes_model.predict_proba(input_data)[0][1]
            
            # Risk factors (enhanced from your second code)
            risks = []
            if glucose > 140: risks.append(f"High glucose ({glucose} mg/dL)")
            if bmi > 30: risks.append(f"Obese (BMI: {bmi:.1f})")
            if age > 45: risks.append(f"Age > 45 ({age} years)")
            
            # Display results (enhanced format)
            st.subheader("Results")
            if proba >= 0.3:
                st.error(f"**Prediction:** Diabetic Detected (Probability: {proba:.1%})")
            else:
                st.success(f"**Prediction:** Not Diabetic Detected (Probability: {proba:.1%})")
            
            if risks:
                st.warning(f"**Risk Factors:** {', '.join(risks)}")
            else:
                st.info("No significant risk factors identified")
            
            # Recommended action (from your second code)
            if proba < 0.3:
                action = "No action needed [You are Healthy]"
            elif proba < 0.7:
                action = "Primary care consult recommended"
            else:
                action = "Immediate endocrinologist evaluation needed"
            
            st.info(f"**Recommended Action:** {action}")
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# ====================== HEART DISEASE PREDICTION (UNCHANGED) ======================
elif selected == 'Heart Disease Prediction':
    st.markdown("<h1>Heart Disease Prediction using ML</h1>", unsafe_allow_html=True)
    
    # Getting the input data (5 key features as per your model)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cp = st.selectbox('Chest Pain Type', 
                         [0, 1, 2, 3],
                         help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
        
        oldpeak = st.number_input('ST Depression Induced by Exercise', 
                                min_value=0.0, max_value=6.2, value=1.0, step=0.1)
    
    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved', 
                                 min_value=60, max_value=220, value=150)
        
        ca = st.selectbox('Number of Major Vessels', 
                         [0, 1, 2, 3, 4])
    with col3:
        exang = st.selectbox('Exercise Induced Angina', 
                            [0, 1],
                            help="0: No, 1: Yes")
    
    # Load scaler from heart model
    if heart_model and isinstance(heart_model, dict) and 'scaler' in heart_model:
        scaler = heart_model['scaler']
    else:
        scaler = StandardScaler()  # Fallback
    
    if st.button('Heart Disease Test Result'):
        try:
            input_data = [[cp, thalach, exang, oldpeak, ca]]
            input_scaled = scaler.transform(input_data)
            
            # Get prediction and probability
            prediction = heart_model['model'].predict(input_scaled)[0]
            proba = heart_model['model'].predict_proba(input_scaled)[0][1]
            
            # Clinical interpretation
            risks = []
            if cp >= 2: risks.append(f"Chest pain type {cp}")
            if thalach < 140: risks.append(f"Low HR ({thalach}bpm)")
            if exang == 1: risks.append("Exercise angina")
            if oldpeak > 1: risks.append(f"ST depression ({oldpeak}mm)")
            if ca > 0: risks.append(f"{ca} blocked vessel(s)")
            
            # Display results
            st.subheader("Results")
            if prediction == 1:
                st.error(f'Heart Disease Detected (Probability: {proba:.1%})')
            else:
                st.success(f'No Heart Disease Detected (Probability: {proba:.1%})')
            
            if risks:
                st.warning("Risk Factors: " + ", ".join(risks))
            
            # Recommended action
            if proba < 0.3:
                action = "Routine check recommended [You are Healthy]"
            elif proba < 0.7:
                action = "Stress test recommended"
            else:
                action = "Immediate angiogram recommended"
            st.info(f"Recommended Action: {action}")
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# ====================== PARKINSONS PREDICTION (UNCHANGED) ======================
elif selected == "Parkinsons Prediction":
    st.markdown("<h1>Parkinson's Disease Prediction using ML</h1>", unsafe_allow_html=True)
    
    # Getting the input data (5 key features as per your model)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        jitter = st.number_input('Jitter (%)', 
                               min_value=0.0, max_value=1.0, value=0.005, step=0.001, format="%.3f")
        
        ppe = st.number_input('PPE (Pitch Period Entropy)', 
                             min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    
    with col2:
        shimmer = st.number_input('Shimmer (dB)', 
                                min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3f")

        hnr = st.number_input('HNR (Harmonic-to-Noise Ratio)', 
                             min_value=0.0, max_value=40.0, value=20.0, step=0.1)
        
    with col3:
        rpde = st.number_input('RPDE (Recurrence Period Density Entropy)', 
                              min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    # Load scaler from Parkinson's model
    if parkinsons_model and isinstance(parkinsons_model, dict) and 'scaler' in parkinsons_model:
        scaler = parkinsons_model['scaler']
    else:
        scaler = StandardScaler()  # Fallback
    
    if st.button("Parkinson's Test Result"):
        try:
            input_data = [[jitter, shimmer, hnr, rpde, ppe]]
            input_scaled = scaler.transform(input_data)
            
            # Get prediction and probability
            prediction = parkinsons_model['model'].predict(input_scaled)[0]
            proba = parkinsons_model['model'].predict_proba(input_scaled)[0][1]
            
            # Clinical interpretation
            risks = []
            if jitter > 0.005: risks.append(f"High jitter ({jitter*100:.2f}%)")
            if shimmer > 0.05: risks.append(f"High shimmer ({shimmer:.2f} dB)")
            if hnr < 20: risks.append(f"Low HNR ({hnr:.1f})")
            if rpde > 0.5: risks.append(f"High RPDE ({rpde:.2f})")
            if ppe > 0.2: risks.append(f"High PPE ({ppe:.2f})")
            
            # Display results
            st.subheader("Results")
            if prediction == 1:
                st.error(f"Parkinson's Detected (Probability: {proba:.1%})")
            else:
                st.success(f"No Parkinson's Detected (Probability: {proba:.1%})")
            
            if risks:
                st.warning("Risk Factors: " + ", ".join(risks))
            
            # Recommended action
            if proba < 0.3:
                action = "No action needed [You are Healthy]"
            elif proba < 0.7:
                action = "Neurologist consult recommended"
            else:
                action = "Immediate specialist evaluation needed"
            st.info(f"Recommended Action: {action}")
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Health Assistant - Multi-Disease Prediction System</p>
    </div>
""", unsafe_allow_html=True)