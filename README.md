<div align="center">

# 🧠 Mental Wellness Analysis and Support Strategy

### Machine Learning Approach to Understanding Mental Health in Tech

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit)](https://openlearn-capstoneproject-188nmv.streamlit.app/)
[![Medium](https://img.shields.io/badge/Medium-Technical_Report-000000?style=for-the-badge&logo=medium)](https://medium.com/@javin.chutani/mental-health-in-tech-a-machine-learning-approach-to-understanding-and-predicting-support-needs-35e86f5bc86f)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Available-2496ED?style=for-the-badge&logo=docker)](https://hub.docker.com/r/javin1106/188nmw)

**A Data Science Project by Javin Chutani**

</div>

---

## 👨‍💻 About

**Author:** Javin Chutani  
**Project Type:** Machine Learning & Data Analytics  
**Status:** Active

---

## 📖 Overview

This project explores mental health challenges faced by employees in the tech industry using machine learning and data analytics. By analyzing survey data from tech workers, the project develops predictive models and insights to help organizations create better mental health support systems.

### 🎯 Key Objectives

1. **Classification Task** - Predict whether an individual is likely to seek mental health treatment based on workplace and personal factors
2. **Regression Task** - Predict age of individuals to design age-targeted interventions
3. **Clustering Analysis** - Segment tech employees into distinct groups based on mental health indicators for tailored HR policies

---

## ✨ Features

- 🔍 **Exploratory Data Analysis (EDA)** - Comprehensive visualization and statistical analysis
- 🤖 **Machine Learning Models**
  - Random Forest Classifier
  - XGBoost Classifier
  - Logistic Regression
  - Random Forest Regressor
  - K-Means Clustering
- 📊 **Interactive Dashboard** - Streamlit web application for model predictions and insights
- 📈 **Model Performance Metrics** - ROC curves, confusion matrices, and detailed evaluation
- 🎨 **Data Visualizations** - Univariate, bivariate, and multivariate analysis
- 🐳 **Docker Support** - Containerized deployment for easy setup

---

## 📂 Project Structure

```
188nmv/
│
├── 📁 Images/                          # Visualization outputs
│   ├── bivariate1.png
│   ├── bivariate2.png
│   ├── cluster0.png
│   ├── cluster1.png
│   ├── cluster2.png
│   ├── cluster3.png
│   ├── dimred.png
│   ├── multivariate1.png
│   ├── multivariate2.png
│   ├── ROC Curve - Classification.png
│   ├── univariate1.png
│   └── univariate2.png
│
├── 📁 Models & Dataset/                # Trained models and processed data
│   ├── classification_model.pkl
│   ├── regression_model.pkl
│   └── df.pkl
│
├── 📁 Notebooks/                       # Jupyter notebooks for analysis
│   ├── EDA.ipynb                       # Exploratory Data Analysis
│   ├── classification_model.ipynb      # Classification model training
│   ├── regression_model.ipynb          # Regression model training
│   └── clustering.ipynb                # Clustering analysis
│
├── 📁 .devcontainer/                   # Development container config
│   └── devcontainer.json
│
├── 📄 app.py                           # Streamlit web application
├── 📄 survey.csv                       # Raw dataset
├── 📄 requirements.txt                 # Python dependencies
├── 📄 Dockerfile                       # Docker configuration
├── 📄 .dockerignore                    # Docker ignore file
└── 📄 README.md                        # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/javin1106/188nmv.git
   cd 188nmv
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

### Option 2: Using Docker

#### Pull from Docker Hub (Recommended)
```bash
# Pull the latest image
docker pull javin1106/188nmw:latest

# Run the container
docker run -p 8501:8501 javin1106/188nmw:latest
```

#### Or Build Locally
```bash
# Build the image
docker build -t mental-wellness-app .

# Run the container
docker run -p 8501:8501 mental-wellness-app
```

#### Docker Compose (if available)
```bash
docker-compose up
```

**Access the application:**
- Open your browser and navigate to `http://localhost:8501`

---

## 📊 Dataset

**Source:** [Mental Health in Tech Survey - Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

The dataset contains responses from tech employees regarding:
- Demographics (age, gender, country)
- Work environment characteristics
- Mental health history
- Workplace mental health benefits
- Attitudes toward mental health treatment

---

## 🧪 Methodology

### 1. Data Preprocessing
- Handling missing values
- Feature engineering
- Encoding categorical variables
- Data normalization and scaling

### 2. Exploratory Data Analysis
- Univariate, bivariate, and multivariate analysis
- Correlation analysis
- Distribution plots and statistical summaries

### 3. Model Development

#### Classification Model
- **Target Variable:** Treatment seeking behavior
- **Algorithms:** Logistic Regression, Random Forest, XGBoost
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

#### Regression Model
- **Target Variable:** Age prediction
- **Algorithms:** Linear Regression, Random Forest Regressor
- **Evaluation Metrics:** RMSE, MAE, R² Score

#### Clustering Analysis
- **Algorithm:** K-Means Clustering
- **Purpose:** Segmentation of employees based on mental health patterns

---

## 📈 Results

The models demonstrate strong predictive capabilities in identifying:
- Employees at risk of mental health issues
- Key workplace factors influencing mental wellness
- Distinct employee segments requiring different support strategies

For detailed results and insights, please refer to the [Technical Report](https://medium.com/@javin.chutani/mental-health-in-tech-a-machine-learning-approach-to-understanding-and-predicting-support-needs-35e86f5bc86f).

---

## 🌐 Live Demo

Experience the interactive dashboard: [**Launch App**](https://javin-mental-health-in-tech.streamlit.app/)

---

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Matplotlib & Seaborn** - Data visualization
- **Streamlit** - Web application framework
- **Joblib** - Model serialization
- **Docker** - Containerization and deployment

---

## 📝 Key Insights

- Family history of mental health issues is a strong predictor of treatment seeking
- Remote work policies impact mental wellness differently across demographics
- Company size and benefits significantly influence employee mental health
- Age-specific interventions can improve support program effectiveness

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/javin1106/188nmv/issues).

---

## 📜 License

This project is open source and available for educational and research purposes.

---

## 🙏 Acknowledgements

- **Dataset Source:** [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Kaggle Community** - For making valuable datasets accessible

---

## 📧 Contact

**Javin Chutani**
- GitHub: [@javin1106](https://github.com/javin1106)
- Medium: [@javin.chutani](https://medium.com/@javin.chutani)
- Docker Hub: [javin1106/188nmw](https://hub.docker.com/r/javin1106/188nmw)

---

<div align="center">

Made with ❤️ for improving mental health awareness in tech

**⭐ Star this repository if you find it helpful!**

</div>
