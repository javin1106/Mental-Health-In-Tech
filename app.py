#Importing libraries
import streamlit as st
import pandas as pd
import joblib
import xgboost

# Load data and models
df = joblib.load("Models & Dataset/df.pkl")

clf_model = joblib.load("Models & Dataset/classification_model.pkl")

reg_model = joblib.load("Models & Dataset/regression_model.pkl")

# App layout
st.set_page_config(page_title="Mental Health Survey App", layout="wide")


# Footer
def footer():
    st.markdown("---")
    st.markdown("""
    <small>Built with ❤️ by Javin Chutani | 
    [LinkedIn](https://www.linkedin.com/in/javin-chutani/) • 
    [GitHub](https://github.com/javin1106) • 
    [X](https://x.com/JavinChutani)</small>
    """, unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🏁 Exploratory Data Analysis",
        "📈 Regression Task",
        "🧮 Classification Task",
        "📊 Persona Clustering"
    ]
)

# 🏠 Home
if menu == "🏠 Home":
    st.title("Mental Health in Tech Survey")
    st.divider()
    st.header("Dataset Overview")
    st.markdown("""
    ### Dataset Source: [Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
    ### Collected by OSMI (Open Sourcing Mental Illness)
    ### Features include:
    * Demographic details (age, gender, country)
    * Workplace environment (mental health benefits, leave policies)
    * Personal experiences (mental illness, family history)
    * Attitudes towards mental health
    """)

    st.header("Problem Statement")
    st.markdown("""
        As a Machine Learning Engineer at NeuronInsights Analytics, you've been contracted by a coalition of
        leading tech companies including CodeLab, QuantumEdge, and SynapseWorks. Alarmed by rising burnout,
        disengagement, and attrition linked to mental health, the consortium seeks data-driven strategies to
        proactively identify and support at-risk employees. Your role is to analyze survey data from over 1,500 tech
        professionals, covering workplace policies, personal mental health history, openness to seeking help, and
        perceived employer support.
                    
        ### Project Objectives:
        * **Exploratory Data Analysis**
        * **Supervised Learning**:
            * *Classification task*: Predict whether a person is likely to seek mental health treatment (treatment column: yes/no)
            * *Regression task*: Predict the respondent's age
        * **Unsupervised Learning**: Cluster tech workers into mental health personas
        * **Streamlit App Deployment**
    """)

    footer()

# 🏁 Data Visualisation
elif menu == "🏁 Exploratory Data Analysis":
    st.title("📊 Data Analysis, Observations & Inferences")
    st.divider()
    st.write("This dataset had many anomalies, null values, outliers, and imbalanced data in columns like `Gender`, `Age`, `Country`, etc. which needed to be cleaned" \
    "and standardised.")
    st.write("Total number of values: `1259`")
    st.write("Total number of features: `27`")
    st.write("Features having NaN values: \n")
    st.write("\t`state`: 451")
    st.write("\t`self_employed`: 18")
    st.write("\t`work_interference`: 264")
    st.write("\t`comments`: 1095")
    st.divider()
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    st.divider()
    removed_features = ['Timestamp', 'Country', 'state', 'self_employed', 'phys_health_consequence', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical',
                        'comments']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Features Used:")
        for col in df.columns:
            st.markdown(f"- {col}")

    with col2:
        st.markdown("### ❌ Features Removed:")
        for col in removed_features:
            st.markdown(f"- {col}")

    st.divider()
    st.header("Univariate Analysis")
    st.image("Images/univariate1.png", caption="Univariate Analysis (1)", use_container_width=True)    

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ###
        - Mean Age of Respondents: `32.05`
        - Top 5 Countries by Responses: `United States`, `United Kingdom`, `Canada`, `Germany`, `Ireland`
        """)

    with col2:
        st.markdown("""
        ### 
        - Male: `79.6%`
        - Female: `19.0%`
        - Other: `1.4%`
        """)

    st.image("Images/univariate2.png",  caption="Univariate Analysis (2)", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        • The dataset shows a nearly balanced split between those who sought treatment (51.6%) and those who didn’t (48.4%).
 
        • Around 39% of respondents have a family history of mental illness, suggesting a potential risk factor.
                    
        • Most respondents reported that mental health *sometimes* interferes with work, while fewer experienced frequent interference.
        """)

    with col2:
        st.markdown("""
        • A strong majority feel comfortable discussing mental health with coworkers, which is a positive sign of workplace openness.

        • Many respondents are unsure about mental health leave policies, showing poor communication or unclear HR policies.

        • Although a majority report having benefits, a large proportion either don’t know or lack them, which is a sign of awareness lack.
        """)

    st.divider()
    st.header("Bivariate Analysis")
    st.image("Images/bivariate1.png", caption="Bivariate Analysis (1)", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        • Females and Others are more likely to seek treatment compared to Males.
 
        • Treatment seeking behavior varies globally with countries like France and Germany having notably lower treatment rates.
        """)

    with col2:
        st.markdown("""
        • People in their late 20s to early 30s form the bulk of those seeking treatment.

        • Those who report frequent work interference are far more likely to seek mental health treatment.
        """)
    st.image("Images/bivariate2.png", caption="Bivariate Analysis (2)", use_container_width=False)
    st.markdown("""
    ###   
    • Most respondents fall between `25–35 years`,  
    • Males are slightly more concentrated in the early 30s,  
    • Females peak in the late 20s,  
    • `Other` gender group is significantly smaller but follows a similar trend.
    """)
    st.divider()

    st.header("Multivariate Analysis")
    st.image("Images/multivariate2.png", caption="Multivariate Analysis (1)", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        • Males dominate the dataset and are **less likely** to seek treatment.  
        • A higher proportion of females **do seek** mental health treatment.
        
        • Most males and females report **"Sometimes"** work interference.  
        • The **"Other"** gender group is minimal but follows similar patterns.
        """)

    with col2:
        st.markdown("""
        • Average age is fairly stable across company sizes (~31–34 years).  
        • People at **large companies** who sought treatment tend to be older.

        • **26–35** is the most common age group for both treated & untreated.  
        • **Significant drop** in treatment observed for respondents **45+**.
        """)

    st.divider()
    st.header("Correlation Heatmap") 
    st.image("Images/multivariate1.png", caption="Correlation Heatmap", use_container_width=True)

    footer()

    
# 📈 Regression
elif menu == "📈 Regression Task":

    st.title("📈 Regression Task")
    st.divider()
    st.markdown("The regression task aims to predict the `Age` of employees using various regression models. Below are the models used, their tuned hyperparameters, and evaluation results.")

    # 📌 Overview of Models
    st.subheader("📌 Models Trained")
    st.write(" - Linear Regression\n - Random Forest Regression\n - XGBoost Regression")

    #Linear Regression
    st.markdown("""### **Linear Regression**""")
    st.write("Linear Regression is a simple and interpretable model that fits a linear relationship between the features and the target variable.")
    st.code("MAE: 5.5234\n" \
    "RMSE: 7.1290\n" \
    "R² Score: 0.0073")

    # Random Forest Regression
    st.markdown("""### **Random Forest Regression**""")
    st.write("Random Forest Regressor is an ensemble method that builds multiple decision trees and averages their predictions for better accuracy and robustness.")
    st.write("Tuned Hyperparameters: \n")
    st.write(" - max_depth: 10\n - min_samples_leaf: 2\n - min_samples_split: 5\n - n_estimators: 200")
    st.code("MAE: 5.2341\n" \
    "RMSE: 6.8873\n" \
    "R² Score: 0.0745")

    # XGB Regression
    st.markdown("""### **XGBoost Regression**""")
    st.write("XGBoost is a powerful gradient boosting algorithm that often achieves state-of-the-art results on structured data.")
    st.write("Tuned hyperparameters: \n")
    st.write(" - col_sample_bytree: 1.0\n - learning_rate: 0.01\n - max_depth: 3\n - n_estimators: 200 - subsample: 0.8")
    st.code("MAE: 5.2612\n" \
    "RMSE: 6.8626\n" \
    "R² Score: 0.0801")

    # Comparison of models
    data = {
        "Model": ["Linear Regression", "Random Forest Regression", "XGBoost Regression"],
        "MAE": [5.5234, 5.2341, 5.2612],
        "RMSE": [7.1290, 6.8873, 6.8626],
        "R² Score": [0.0073, 0.0745, 0.0802]
    }
    results_df = pd.DataFrame(data)
    results_df = results_df.sort_values(by="R² Score", ascending=False)
    st.divider()
    st.markdown("## 📊 Regression Model Comparison")
    st.markdown("A summary of model performance metrics:")

    st.dataframe(results_df.style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "R² Score": "{:.4f}"
    }), use_container_width=True)

    st.divider()
    st.markdown("### ✅ By looking at the evaluation metrics, we can see that `XGB Regression` performs the best and will be used for prediction purposes.")
    
    st.divider()
    st.markdown("### This is a sample predictor of the age of a person given their conditions ⬇️")

    input_dict_reg = {}
    display_names_reg =  {
    "self_employed": "Are you self-employed?",
    "Gender": "Enter your Gender",
    "family_history": "Do you have a family history of mental illness?",
    "treatment": "Have you sought treatment for a mental health condition?",
    "work_interfere": "If you have a mental health condition, do you feel that it interferes with your work?",
    "no_employees": "What is the size of your company by number of employees?",
    "remote_work": "Do you work remotely (outside of an office) at least 50% of the time?",
    "benefits": "Does your employer provide mental health benefits?",
    "care_options": "Do you know the options for mental health care your employer provides?",
    "wellness_program": "Has your employer ever discussed mental health as part of an employee wellness program?",
    "seek_help": "Does your employer provide resources to learn more about mental health issues and how to seek help?",
    "anonymity": "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?",
    "leave": "How easy is it for you to take medical leave for a mental health condition?",
    "mental_health_consequence": "Do you think that discussing a mental health issue with your employer would have negative consequences?",
    "coworkers": "Would you be willing to discuss a mental health issue with your coworkers?",
    "supervisor": "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
    "obs_consequence": "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",
    }

    for col in df.columns:
        if col == "Age" or col == "age_group":  
            continue

        options = df[col].dropna().unique().tolist()
        label = display_names_reg.get(col, col) 
        input_dict_reg[col] = st.selectbox(label, options)

    # Convert inputs into DataFrame
    input_df = pd.DataFrame([input_dict_reg])

    # Predict Age
    if st.button("Predict Age"):
        # transformed = reg_pre.transform(input_df)
        predicted_age = reg_model.predict(df)

        st.success(f"🎯 Predicted Age: **{int(round(predicted_age[0]))} years**")

    footer()

# 🧮 Classification
elif menu == "🧮 Classification Task":
    st.title("🧮 Will the person seek treatment?")
    st.divider()
    st.markdown("The task at hand is to estimate wether the employee would seek help or not, making it a binary classification task. Below are the models used, and their evaluation results")
    
    # 📌 Overview of Models
    st.subheader("📌 Models Trained")
    st.write(" - Logistic Regression\n - Random Forest Classification\n - KNN Classification\n - XGB Classification")

    # Logistic Regression
    st.markdown("""### **Logisitc Regression**""")
    st.write("A baseline linear classifier that predicts probabilities and works well for binary treatment prediction. The hyperparameters were tuned to reduce overfitting.")
    st.write("Tuned hyperparamters: \n")
    st.write(" - C: 1\n - penalty: l2\n - solver: lbfgs")
    st.code("Accuracy: 0.8128\n ROC-AUC Score: 0.8897\n F1 Score: 0.84")

    # Random Forest Classification
    st.markdown("""### **Random Forest Classification**""")
    st.write("An ensemble model that uses multiple decision trees to improve classification stability and accuracy. Cross validation using `GridSearchCV` has been used to reduce overfitting.")
    st.write("Tuned hyperparamters: \n")
    st.write(" - max_depth: 5\n - max_features: log2\n - n_estimators: 100")
    st.code("Accuracy: 0.8181\n ROC-AUC Score: 0.9039\n F1 Score: 0.84")

    # KNN Classification
    st.markdown("""### **KNN Classification**""")
    st.write("Classifies based on the majority label among the closest data points in feature space.")
    st.write("Tuned hyperparameters: \n")
    st.write(" - metric: manhattan\n - n_neighbors: 10\n - weights: distance")
    st.code("Accuracy: 0.7807\n ROC-AUC Score: 0.8677\n F1 Score: 0.79")

    # XGB Classification
    st.markdown("""### **XGB Classification**""")
    st.write("A powerful boosting-based classifier known for high accuracy, especially on structured and tabular data.")
    st.write("Tuned hyperparameters: \n")
    st.write(" - learning_rate: 0.01\n - max_depth: 3\n - n_estimators: 300")
    st.code("Accuracy: 0.8288\n ROC-AUC Score: 0.9035\n F1 Score: 0.86")

    # Comparison of models
    data = {
        "Model": ["Logistic Regression", "Random Forest Classification","KNN Classification", "XGBoost Classification"],
        "Accuracy": [0.8128, 0.8181, 0.7807, 0.8288],
        "ROC-AUC Score": [0.8897, 0.9039, 0.8677, 0.9035],
        "F1 Score": [0.84, 0.84, 0.79, 0.86]
    }
    results_clf_df = pd.DataFrame(data)
    results_clf_df = results_clf_df.sort_values(by="ROC-AUC Score", ascending=False)
    st.divider()
    st.markdown("## 📊 Classification Model Comparison")
    st.markdown("A summary of model performance metrics:")

    st.dataframe(results_clf_df.style.format({
        "Accuracy": "{:.4f}",
        "ROC-AUC Score": "{:.4f}",
        "F1 Score": "{:.4f}"
    }), use_container_width=True)

    st.image("Images/ROC Curve - Classification.png", caption="ROC Curve for different models", use_container_width=False)

    st.divider()
    st.markdown("### ✅ From the evaluation metrics it is evident that `XGB Classifiction` wins the race marginally over `Random Forest Classification`.")

    st.divider()
    st.markdown("### This is a sample predictor whether a person with given conditions is likely to seek mental health support or not ⬇️")

    input_dict_clf = {}
    display_names_clf = {
    "self_employed": "Are you self-employed?",
    "Gender": "Enter your Gender",
    "family_history": "Do you have a family history of mental illness?",
    "work_interfere": "If you have a mental health condition, do you feel that it interferes with your work?",
    "no_employees": "What is the size of your company by number of employees?",
    "remote_work": "Do you work remotely (outside of an office) at least 50% of the time?",
    "benefits": "Does your employer provide mental health benefits?",
    "care_options": "Do you know the options for mental health care your employer provides?",
    "wellness_program": "Has your employer ever discussed mental health as part of an employee wellness program?",
    "seek_help": "Does your employer provide resources to learn more about mental health issues and how to seek help?",
    "anonymity": "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?",
    "leave": "How easy is it for you to take medical leave for a mental health condition?",
    "mental_health_consequence": "Do you think that discussing a mental health issue with your employer would have negative consequences?",
    "coworkers": "Would you be willing to discuss a mental health issue with your coworkers?",
    "supervisor": "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
    "obs_consequence": "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",
    }

    for col in df.columns:
        if col == "age_group":  # Skip this column
            continue

        if col == "treatment": 
            continue

        if col == "Age":
            input_dict_clf[col] = st.number_input("Enter your age", min_value=19, max_value=100, step=1)
        else:
            options = df[col].dropna().unique().tolist()
            label = display_names_clf.get(col, col)
            input_dict_clf[col] = st.selectbox(label, options)

    input_df = pd.DataFrame([input_dict_clf])

    # Predict 
    if st.button("Predict"):
        input_df = pd.DataFrame([input_dict_clf]) 
        prediction = clf_model.predict(input_df)[0]

        # Step 4: Output result
        if prediction == 1:
            st.success("✅ Predicted: Will likely seek treatment!")
        else:
            st.error("❌ Predicted: Will likely not seek treatment!")

    footer()

# 📊 Clustering
elif menu == "📊 Persona Clustering":
    st.title("📊 Clustering Analysis")
    st.divider()
    st.markdown("The objective of this task is to make clusters and group tech workers according to their mental health personas. Below are some of the techniques and algorithms applied for the same.")
    st.write("The columns `Age`, `Country`, `Gender`, `no_employees`, `wellness_program`, `care_options`, `mental_health_consequence`, `benefits` were dropped due to their less contribution to the overall cluster making. These features" \
    "somewhere get covered in the rest of the questionnaire filled by the respondents.")

    # Clustering techniques
    st.subheader("Techniques Used: ")
    st.write(" - Principal Component Analysis (PCA)\n - t-distributed Stochastic Neighbor Embedding (t-SNE)\n - Uniform Manifold Approximation and Projection (UMAP)")
    st.write("Here is the plot for all three techniques applied on this dataset.")
    st.image("Images/dimred.png", caption="From these clusters we can see that `UMAP` forms the best and most clear and seggregated clusters out of the three.", use_container_width=True)
    
    st.write("The most optimal number of clusters were found to be 6, ranked by silhouette score for each cluster number.")

    st.divider()

    st.markdown("### Here is the comparison (silhouette score) for all the models applied for clustering: ")
    st.write(" - **K-Means Clustering:** 0.4836\n - **Agglomerative Clustering:** 0.4619\n - **DBSCAN:** 0.2192 (13 DBSCAN Clusters, 2 Noise Points)")

    st.markdown("### ✅ From these scores, we can easily say that `K-Means` is clearly our winner.")
    st.image("Images/clusters.png", caption="Clusters formed by the models", use_container_width=True)

    st.divider()

    st.markdown("### 🧠 These are the different Personas the respondents can be classified into ⬇️")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌱 Cluster 1", "🔥 Cluster 2", "😐 Cluster 3",
    "📢 Cluster 4", "🚫 Cluster 5", "⚖️ Cluster 6"
    ])

    with tab1:
        st.markdown("""
        ### 🌱 Cluster 1: Stable but Unaware Support-Seekers  
        Mostly aged 26–35, this group feels mentally steady at work and is comfortable talking to coworkers and supervisors.  
        However, they lack clarity around workplace policies like **mental health leave**, **anonymity**, and **support access**.  
        They're socially supportive and open, but not actively engaged with mental health systems or formal resources.  
        Their attitude is positive, but their actions show low involvement in structured support mechanisms.
        """)

    with tab2:
        st.markdown("""
        ### 🔥 Cluster 2: Distressed but Expressive Fighters  
        These individuals are actively struggling many receive treatment, face regular **work interference**, and find it hard to take leave.  
        Still, they are emotionally open and communicate with supervisors and peers.  
        There's a clear lack of awareness about workplace policies, and some might be receiving help not by choice, but through family or HR intervention.  
        They are courageous and expressive, but often stuck in **rigid, unsupportive work environments**.
        """)

    with tab3:
        st.markdown("""
        ### 😐 Cluster 3: Detached Optimists  
        This group doesn’t feel affected by mental health concerns or may not recognize them at all.  
        They haven’t received treatment, don’t understand workplace mental health policies, and are unsure about seeking help.  
        Their optimism may stem from **limited exposure**, **cultural silence**, or **low awareness**.  
        While they’re somewhat socially open, they remain **uninformed and passive** when it comes to mental health.
        """)

    with tab4:
        st.markdown("""
        ### 📢 Cluster 4: Vocal Realists  
        These individuals acknowledge that mental health can interfere with work and many have sought treatment.  
        They benefit from **supportive colleagues and supervisors**, encouraging open dialogue in the workplace.  
        However, they’re still navigating uncertain ground around **anonymity**, **leave policies**, and formal procedures.  
        They are mentally aware, confident, and gradually pushing for cultural shifts in workplace norms.
        """)

    with tab5:
        st.markdown("""
        ### 🚫 Cluster 5: Detached Dismissers  
        This group largely **denies** mental health impacts at work and shows little interest in seeking help.  
        They’re mostly unaware of leave or support policies and have rarely, if ever, received treatment.  
        Although some receive **mild support** from peers, engagement with mental health systems is minimal.  
        Predominantly younger professionals, they appear disconnected or indifferent to the broader mental health conversation.
        """)

    with tab6:
        st.markdown("""
        ### ⚖️ Cluster 6: Conflicted Strugglers  
        Many here have received treatment before, but currently hesitate to seek help possibly due to burnout, skepticism, or distrust.  
        They’re confused about policies like **anonymity** and **leave benefits**, and their workplace support is inconsistent.  
        Some coworkers and supervisors are helpful, while others aren’t.  
        This group is mentally aware but emotionally drained, cautious, and trying to find stable ground in a mixed support system.
        """)

    footer()
