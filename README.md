# ABC Tech ITSM Incident Analysis & Predictive Automation

## 🎯 Project Overview

This project delivers a robust, end-to-end data science pipeline built to analyze historical IT Service Management (ITSM) incident data from a major tech firm, **ABC Tech**, and deploy predictive models for operational optimization.

The primary objective was to move ITSM operations from reactive to proactive by automating triage and forecasting future resource needs.

---

## ✨ Key Achievements & Deliverables

The project successfully addressed all four key client objectives, resulting in quantifiable improvements to efficiency and risk management. The project utilized a **45,000+ record, SQL-sourced ITSM dataset**.

| Goal | Description | Key Achievement |
| :--- | :--- | :--- |
| **1. High Priority Prediction** | Forecast whether an incoming ticket will be high priority (P1/P2). | **96% Accuracy** and **0.89 AUC** achieved by resolving critical data leakage. |
| **2. Incident Forecasting** | Forecast incident volume across quarterly and annual horizons. | **Stabilized forecasting models** (Exponential Smoothing, Regression) to provide **reliable, non-negative volume projections** for staffing and capacity planning. |
| **3. Auto-Tagging** | Automatically assign the correct **Priority (P1-P5)** and **Department (CI\_Cat)**. | **75% Accuracy** in multi-class Department Tagging and high-confidence Priority routing. |
| **4. RFC Failure Prediction** | Predict the likelihood of a Request-for-Change (RFC) leading to a failure or misconfiguration. | **98% Test F1-Score** achieved using a sophisticated XGBoost model with SMOTE, enabling proactive risk mitigation. |

---

## ⚙️ Technical Stack

*   **Language:** Python (3.x)
*   **Data Handling:** `Pandas`, `NumPy`, `MySQL Connector`
*   **Machine Learning:** `Scikit-learn` (Pipelines, ColumnTransformer, RandomForest), `XGBoost`
*   **Time Series:** `statsmodels` (Exponential Smoothing, Holt), `Scikit-learn` (Linear Regression)
*   **Imbalance Handling:** `imblearn` (SMOTE)
*   **Visualization:** `Matplotlib`, `Seaborn`
*   **Deployment Assets:** `joblib` (for saving trained models and scalers)

---

## 🛠️ Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd abc-tech-itsm-analysis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    .\venv\Scripts\activate    # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn mysql-connector-python imbalanced-learn statsmodels
    ```

4.  **Database Connection:**
    The notebook connects to a MySQL database using internal credentials. If you are running this locally, ensure you update the database connection details in the **Data Acquisition & Preprocessing** cell of the notebook.

---

## 🚀 Execution

The core of the project is contained within the `ABC_TECH_ITSM_Project.ipynb` notebook.

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Run all cells sequentially. The notebook executes the following steps:
    *   **Preprocessing:** Robust data cleaning, imputation, scaling, and one-hot encoding.
    *   **Goal 1:** High Priority Incident Prediction.
    *   **Goal 2:** Incident Volume Forecasting (Quarterly, Annual, Monthly).
    *   **Goal 3:** Department Auto-Tagging.
    *   **Goal 4:** RFC Failure Prediction.

---

## 📂 Repository Structure

## 📂 Repository Structure

```text
├── ABC_TECH_ITSM_Project.ipynb  # Main Project Notebook
├── README.md                    # This file
└── Saved_Models/
    ├── priority_binary_rf.pkl   # Model for Goal 3.1 (Priority)
    ├── goal3_department_model_safe.pkl # Pipeline for Goal 3.2 (CI_Cat)
    └── ...                      # Other saved model artifacts
