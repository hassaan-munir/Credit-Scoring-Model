# Credit Scoring Model

## Project Overview

This project involves building a Machine Learning model to assess credit risk based on various financial attributes. The primary goal is to predict the likelihood of a loan default, enabling financial institutions to make more informed lending decisions. This project was completed as part of the **CodeAlpha Machine Learning Internship**.

## Dataset

The model was developed using a comprehensive credit risk dataset. This dataset includes crucial features such as:

  * `person_age`
  * `person_income`
  * `person_emp_length` (employment length)
  * `loan_amnt` (loan amount)
  * `loan_int_rate` (loan interest rate)
  * `loan_percent_income` (loan amount as a percentage of income)
  * `cb_person_cred_hist_length` (credit history length)
  * Various categorical features related to `person_home_ownership`, `loan_intent`, and `loan_grade`.

The target variable is `loan_status`, indicating whether the loan was defaulted (1) or not (0).

## Methodology

The development process followed a standard Machine Learning pipeline:

1.  **Data Preprocessing:**
      * Handled missing values in `person_emp_length` and `loan_int_rate` by imputing them with the median.
      * Converted categorical features (e.g., `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`) into numerical format using One-Hot Encoding.
2.  **Exploratory Data Analysis (EDA):** Initial checks (`.head()`, `.info()`, `.describe()`) were performed to understand the dataset structure, data types, and basic statistics.
3.  **Model Training:**
      * The preprocessed data was split into training and testing sets (75% training, 25% testing).
      * Classification models such as **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier** were implemented and trained on the training data.
4.  **Model Evaluation:**
      * Models were evaluated on the unseen test set using a range of metrics:
          * Accuracy
          * Precision
          * Recall
          * F1-Score
          * ROC-AUC Score
      * A **Confusion Matrix** was generated to visualize the model's performance in terms of true positives, true negatives, false positives, and false negatives.

## Key Results

The **Random Forest Classifier** consistently demonstrated the most robust performance among the models tested. It achieved an accuracy of approximately **[Apni best accuracy yahan likho, maslan 85-90%]** on the test set, showing strong capability in predicting loan default status. The model effectively balanced precision and recall across both classes.

## Technologies Used

  * **Python**: The primary programming language.
  * **Pandas**: For data manipulation and analysis.
  * **NumPy**: For numerical operations.
  * **Scikit-learn**: For machine learning model implementation, data splitting, and evaluation metrics.
  * **Matplotlib**: For data visualization.
  * **Seaborn**: For enhanced statistical data visualization.

## How to Run

To run this project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/Credit-Scoring-Model.git
    cd Credit-Scoring-Model
    ```
    *(Replace `YourUsername` with your actual GitHub username and `Credit-Scoring-Model` with your repo name if different)*
2.  **Download the dataset:** Ensure the `credit_risk_dataset.csv` file is in the same directory as the Jupyter Notebook.
3.  **Install required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
4.  **Open the Jupyter Notebook:**
    Launch Jupyter Notebook (or open in VS Code with Jupyter extension) and open the `[Your_Notebook_Name].ipynb` file (e.g., `credit_scoring_model.ipynb`).
5.  **Run all cells:** Execute all cells in the notebook sequentially to preprocess data, train models, and evaluate their performance.

## Internship

This project was completed as a core task during my **Machine Learning Internship at CodeAlpha**.

## Connect with Me

   **Muhammad Hassaan Munir** ( Linkedin - https://www.linkedin.com/in/muhammad-hassaan-munir-79b5b2327/)

