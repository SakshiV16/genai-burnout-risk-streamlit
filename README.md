# GenAI Usage and Student Burnout Risk: Five-Model Classification Comparison

**Machine Learning – Assignment 2**

**Live Streamlit App:** [https://genai-burnout-risk-app.streamlit.app/](https://genai-burnout-risk-app.streamlit.app/)

## 1. Problem Statement

Generative AI tools are now part of everyday student life, and heavy use may be linked to wellbeing outcomes such as burnout. The objective of this project is to **predict a student's burnout risk level (Low, Medium or High)** from academic background, GenAI usage, study behavior and institutional factors.

This is a **multi-class classification** problem. Five machine learning models (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes and Support Vector Machine) are trained on the same data, evaluated on a held-out test set with Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC), and compared to find the best model for this dataset. The project looks for patterns in the data. It does not claim that using AI causes burnout.

## 2. Dataset Description

- **Source:** "Impact of AI on Students" dataset from Kaggle, [https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students)
- **Size:** 50,000 student records and 16 columns. There are no missing values.
- **Target:** `Burnout_Risk_Level` with three classes: Low, Medium, High. The classes are moderately imbalanced (approximately 33% Low, 42% Medium and 25% High).
- **Features:** After removing the `Student_ID` column, **14 input features** remain, which meets the minimum of 12.

| Group | Features | Type |
|---|---|---|
| Academic background | `Major_Category`, `Year_of_Study`, `Pre_Semester_GPA`, `Post_Semester_GPA`, `Skill_Retention_Score` | 2 categorical, 3 numeric |
| GenAI usage | `Weekly_GenAI_Hours`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Tool_Diversity`, `Paid_Subscription`, `Perceived_AI_Dependency` | 3 categorical (including one boolean), 3 numeric |
| Study behaviour and wellbeing | `Traditional_Study_Hours`, `Anxiety_Level_During_Exams` | 2 numeric |
| Institutional | `Institutional_Policy` | 1 categorical |

In total there are 8 numeric and 6 categorical features.

### Instances Used

To keep training and tuning fast (the SVM is slow on 50,000 rows), a **stratified random sample of 10,000 rows** was drawn, so the class proportions stay the same as in the full dataset. This is well above the minimum of 500 instances.

The sample was split 80/20 (stratified, `random_state=42`) into **8,000 training rows and 2,000 test rows**. The 2,000 test rows are saved as `test_data.csv` and are the file used in the Streamlit app.

### Baseline

Always predicting the majority class gives an accuracy of 0.423 (AUC 0.5, MCC 0), so any useful model must beat this.

### Preprocessing

(Inside each saved pipeline):

- Numeric features use median imputation and standard scaling.
- Categorical features use most-frequent imputation and one-hot encoding.

Because the preprocessing is stored with each model, the app can score a raw test CSV directly.

## 3. GitHub Repository Link

[https://github.com/SakshiV16/genai-burnout-risk-streamlit](https://github.com/SakshiV16/genai-burnout-risk-streamlit)

## 4. Models Used

All five models were implemented on the same dataset and the same train/test split. Each model is a scikit-learn pipeline (preprocessing plus classifier). Hyperparameters were chosen with cross-validation on the training set only.

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=1000`, default regularisation |
| Decision Tree | `max_depth=5`, `min_samples_leaf=1` (tuned) |
| KNN | `n_neighbors=21` (tuned) |
| Naive Bayes | Gaussian Naive Bayes, default settings |
| Support Vector Machine | RBF kernel, `C=0.5` (tuned), `probability=True` |

For this three-class problem, Precision, Recall and F1 are **macro-averaged** and AUC is **one-vs-rest, macro-averaged**.

## 5. Comparison Table (Test Set, 2,000 Rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.5420** | **0.7266** | 0.5708 | 0.5305 | **0.5429** | 0.2844 |
| Decision Tree | 0.5345 | 0.6996 | 0.5674 | 0.5164 | 0.5296 | 0.2699 |
| KNN | 0.4975 | 0.6670 | 0.5259 | 0.4891 | 0.5001 | 0.2161 |
| Naive Bayes | 0.5365 | 0.7126 | 0.5486 | **0.5495** | 0.5408 | **0.3018** |
| Support Vector Machine | **0.5455** | 0.7197 | **0.5972** | 0.5189 | 0.5332 | 0.2875 |

## 6. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | The most consistent model. It has the best AUC (0.7266) and best F1 (0.5429), and is within 0.4 points of the best accuracy. It also had the best cross-validated scores of the five models (accuracy 0.529, F1 0.530, MCC 0.263) on the training set. The result suggests that the main signals in this data, such as weekly GenAI hours and perceived AI dependency, can be captured well by a simple linear boundary. |
| Decision Tree | Performance depended heavily on depth. With default settings (an unrestricted tree) its cross-validated accuracy was only 0.430, close to the baseline, because it overfitted. Limiting `max_depth` to 5 lifted test accuracy to 0.5345 and F1 to 0.5296. Its AUC (0.6996) is lower than the models that output smoother probabilities, since a shallow tree gives only a few distinct probability values. It is easy to interpret but not the most accurate. |
| KNN | The weakest model on every metric (accuracy 0.4975, AUC 0.667, MCC 0.2161), only about 7 points above the 0.423 baseline. Distance-based methods suffer when the signal is weak and the one-hot encoding adds many extra dimensions, so neighbors are noisy. The best k was 21, the largest value tried, so slightly larger values might help a little, but it is unlikely to catch the other models. |
| Naive Bayes | A strong result for such a simple model. It has the highest MCC (0.3018) and the highest recall (0.5495), so it handles the three classes in the most balanced way. Its precision (0.5486) is lower than that of SVM or Logistic Regression. Its test MCC is higher than its cross-validated MCC (0.255), so part of its lead may be due to the particular test split. The independence assumption is not strictly true here, but it did not hurt much. |
| Support Vector Machine | Best accuracy (0.5455) and best precision (0.5972), with a high AUC (0.7197). Its recall (0.5189) and F1 (0.5332) are lower than Logistic Regression's, so it is the most conservative model: when it predicts a class it is often right, but it misses more true cases. It is also the slowest model to train and tune. |

## 7. Overall Winner for this Dataset

**Logistic Regression.**

It has the best AUC and F1 on the test set, is second best on accuracy, precision and recall, and was the top model in cross-validation on accuracy, F1 and MCC. No other model is in the top two on that many measures.

Naive Bayes is the best choice if MCC is the main criterion, and the SVM is the best if accuracy or precision matters most.

### Important Caveat

The top models are separated by only about one percentage point on most metrics, which is similar to the fold-to-fold variation seen in cross-validation (about ±1 point on accuracy).

The ranking among Logistic Regression, SVM, Naive Bayes and the tuned Decision Tree should therefore not be treated as a strong difference, while KNN is clearly behind.

All models beat the majority-class baseline (accuracy 0.423) by 7 to 12 points, so the features carry real but moderate signal about burnout risk.

## 8. Limitations

- The data may be synthetic or survey-style, and the signal is only moderate (best accuracy about 55%).
- `Post_Semester_GPA` and `Skill_Retention_Score` are measured at the end of the semester, so they may not be available when predicting burnout in practice.
- Models were trained on a 10,000-row sample of the full 50,000 rows, and evaluated on a single 2,000-row test split.

---

**Live Streamlit App:** [https://genai-burnout-risk-app.streamlit.app/](https://genai-burnout-risk-app.streamlit.app/)

**GitHub Repository:** [https://github.com/SakshiV16/genai-burnout-risk-streamlit](https://github.com/SakshiV16/genai-burnout-risk-streamlit)
