# SMART HEALTH ASSISTANT: Multi-Disease Prediction System

![image](https://github.com/user-attachments/assets/a943a1e9-ddfd-442d-a580-f9f69ed27107)



## 🏥 Project Overview
A machine learning-powered web application that predicts three critical diseases:
- **Diabetes** (Random Forest, 75.32% accuracy)
- **Heart Disease** (Random Forest, 82.38% accuracy) 
- **Parkinson's Disease** (SVM, 87.41% accuracy)


## 🌟 Key Features
- **Unified Prediction Platform**: Single interface for multiple diseases
- **Clinical-Grade Models**: Optimized Random Forest and SVM algorithms
- **Explainable AI**: Feature importance analysis and confidence scores
- **User-Friendly UI**: Intuitive Streamlit web interface
- **Proactive Healthcare**: Early risk detection with actionable insights

## 🛠️ Technical Implementation
### Algorithms
| Disease        | Algorithm      | Accuracy | AUC  |
|----------------|---------------|----------|------|
| Diabetes       | Random Forest | 75.32%   | 0.81 |
| Heart Disease  | Random Forest | 82.38%   | 0.90 |
| Parkinson's    | SVM           | 87.41%   | 0.93 |

### Tech Stack
- **Frontend**: Streamlit (Python)
- **ML Frameworks**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, SHAP (for explainability)

## 📂 Repository Structure
```
SMART-HEALTH-ASSISTANT/
├── models/                   # Pretrained models (.sav)
│   ├── best_diabetes_model.sav
│   ├── best_heart_model.sav
│   └── best_parkinsons_model.sav
├── datasets/                 # Sample datasets
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
├── notebooks/                # Jupyter notebooks
│   ├── MDPS_Diabetes.ipynb
│   ├── MDPS_Heart.ipynb
│   └── MDPS_Parkinson's.ipynb
├── app/                      # Streamlit application
│   └── healthapp.py          # Main application file
└── README.md                 # This file
```

## 🚀 Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/khadeerBasha44/SMART-HEALTH-ASSISTANT.git
   cd SMART-HEALTH-ASSISTANT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app/healthapp.py
   ```

## 📊 Sample Prediction
![image](https://github.com/user-attachments/assets/e8b7ad75-42fd-4aba-b586-621e4978904b)

## 📜 Citation
If you use this work in your research, please cite:
```bibtex
@thesis{khadeer2025health,
  title={SMART HEALTH ASSISTANT: A ML MODEL FOR MULTI-DISEASE PREDICTION},
  author={Khadeer Basha K},
  year={2025},
  school={Vellore Institute of Technology}
}
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Contact
Khadeer Basha K
📧 khadeershaik2906@gmail.com
🔗 LinkedIn
🎓 VIT Vellore (20MIY0044)

---

*Developed with ❤️ for better healthcare accessibility*

---
