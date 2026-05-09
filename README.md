# AI-Based Diabetes Detection using Tongue Image Analysis

A deep learning system that analyzes tongue images to classify
Diabetic vs Non-Diabetic using ResNet50 feature extraction and a
custom-built Radial Basis Function Neural Network (RBFNN).

**FEATURES:**
- ResNet50 feature extraction
- Custom RBF Neural Network implementation
- 5-Fold Cross Validation
- Accuracy: ~95%
- Streamlit Web App
- Confidence scoring
- Tongue image upload interface

**TECH STACK:**
Python, PyTorch, NumPy, Scikit-learn, OpenCV, Streamlit

## PERFORMANCE METRICS

- Accuracy: 95.50%
- Precision: 91.74%
- Recall (Sensitivity): 100%
- Specificity: 91.00%
- F1 Score: 95.69%
- 5-Fold Cross Validation Accuracy: 95.57% ± 1.10%

# How to Run
**1) Clone repository**
git clone https://github.com/Prantik66/TongueVision-AI.git

**2) Move into project folder**
cd TongueVision-AI

**3) Create virtual environment**
python -m venv venv

**4) Activate virtual environment**

**Windows**
venv\Scripts\activate

**Mac/Linux**
source venv/bin/activate

**5) Install dependencies**
pip install -r requirements.txt

**6) Run Streamlit application**
streamlit run app/app.py
