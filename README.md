# 🤖 AutoML Intelligence Platform

> 🚀 A powerful Streamlit-based AutoML web app for automated machine learning (Classification & Regression) with interactive UI, EDA, model comparison, and downloadable predictions.

---

## 🌟 Overview
This project is an **end-to-end Automated Machine Learning (AutoML) platform** built using **Streamlit**. It allows users to upload datasets and automatically perform:

- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Model training & comparison  
- Best model selection  
- Predictions & downloads  

It is designed to make machine learning **easy, interactive, and visually appealing**.

---

## 🎯 Key Features

### 📂 Data Handling
- Upload CSV or Excel files  
- Automatic preprocessing (missing values, encoding, scaling)  

### 📊 Exploratory Data Analysis (EDA)
- Dataset preview  
- Dataset info & summary statistics  
- Correlation heatmap (interactive)  
- Histograms for numeric features  

### 🧠 AutoML Engine
- Auto-detects **Classification / Regression**
- Trains multiple models:
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting  
  - KNN  
  - Decision Tree  
  - Linear & Ridge Regression  

### ⚙️ Customization
- Adjustable test size  
- KNN neighbors  
- Tree depth  
- Number of estimators  

### 📈 Evaluation Metrics
- **Classification:** Accuracy, F1 Score  
- **Regression:** R² Score, MAE  

### 🏆 Smart Model Selection
- Automatically selects best model based on performance  

### 📉 Visualization
- Feature Importance (for tree models)  
- Confusion Matrix (for classification)  

### 💾 Export Results
- Download predictions in:
  - CSV  
  - Excel  
  - JSON  

---

## 🛠️ Technologies Used
- Python 🐍  
- Streamlit 🎨  
- Pandas & NumPy  
- Scikit-learn  
- Plotly (interactive visualizations)  

---

## 📁 Project Structure

AutoML-Platform/
│
├── app.py # Main Streamlit App
├── requirements.txt # Dependencies
├── data/ # Sample datasets
├── outputs/ # Predictions output
└── README.md


---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/automl-platform.git
Go to project folder:
cd automl-platform
Install dependencies:
pip install -r requirements.txt
▶️ Run the App
streamlit run app.py
💡 How to Use
Upload your dataset (CSV/Excel)
Select target column
Choose problem type (Auto / Classification / Regression)
Adjust model parameters
Click "Run AutoML"
View results & download predictions
📊 Example Output
Best Model: Random Forest
Accuracy / R² Score displayed
Visual charts (EDA + Feature Importance)
Downloadable prediction file
🔮 Future Improvements
Hyperparameter tuning (GridSearch / Auto tuning)
Deep Learning integration
Deployment (Streamlit Cloud / HuggingFace)
Real-time prediction API
🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

📜 License

This project is licensed under the MIT License.

🙋‍♀️ Author

Aleeza Bashir
💻 Aspiring Data Scientist
🚀 Passionate about AI & Machine Learning

⭐ Support

If you like this project, don't forget to star ⭐ the repository!
