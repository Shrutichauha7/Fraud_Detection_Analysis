💳 Fraud Detection Analysis using Machine Learning
📌 Project Overview

This project focuses on detecting fraudulent financial transactions using machine learning techniques. The dataset contains 20,000+ transaction records, with a key challenge of highly imbalanced fraud data.

The goal is to build a robust model that maximizes fraud detection (recall) while keeping false positives low, ensuring practical usability in real-world financial systems.

🚀 Key Features
Data preprocessing and handling of imbalanced datasets
Implementation of multiple ML models
Performance comparison using evaluation metrics
Optimization using SMOTE (Synthetic Minority Oversampling Technique)
Final model selection based on real-world fraud detection priorities
🧠 Machine Learning Models Used
Logistic Regression
Random Forest
Random Forest (with SMOTE)
XGBoost ✅ (Final Model)
📊 Evaluation Metrics

To ensure reliable fraud detection, the following metrics were used:

Precision → Minimizing false positives
Recall → Maximizing fraud detection (primary focus)
F1-Score → Balance between precision & recall
Confusion Matrix → Detailed prediction breakdown
ROC-AUC Score → Overall model performance
🏆 Final Model Selection
Selected Model: XGBoost
ROC-AUC Score: 0.97
Achieved highest fraud recall with strong overall performance
Best balance between detecting fraud and reducing false alarms
⚙️ Tech Stack
Python 🐍
Scikit-learn
XGBoost
Pandas, NumPy
Matplotlib, Seaborn
📁 Project Structure
├── data/                # Dataset files
├── notebooks/           # Jupyter notebooks (EDA & modeling)
├── models/              # Saved trained models
├── src/                 # Source code
├── results/             # Evaluation outputs & visualizations
└── README.md            # Project documentation
🔍 Key Insights
Fraud detection requires handling class imbalance effectively
Models without balancing techniques tend to miss fraud cases
SMOTE significantly improves recall
XGBoost outperformed all models in detecting fraudulent transactions
📌 How to Run the Project
# Clone the repository
git clone https://github.com/Shrutichauha7/AI_wallet_with_Fraud_Detection.git

# Navigate to project directory
cd AI_wallet_with_Fraud_Detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
📈 Future Improvements
Deploy model using Flask/FastAPI
Real-time fraud detection pipeline
Hyperparameter tuning for further optimization
Integration with financial transaction APIs
