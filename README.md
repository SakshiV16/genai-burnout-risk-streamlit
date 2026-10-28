# ml-classifier-comparison-streamlit
Five-model classification comparison with a deployed Streamlit dashboard for uploading test data and viewing evaluation metrics.

Objective: To predict the student's burnout risk level (Low, Medium, or High) based on data relating to AI usage and study habits.

Context: Students are using AI. It's important to understand the impact this has on their well-being.

Data: The data is from a Kaggle dataset that contains 50,000 student records. We've removed the student ID, so 14 input features remain.

Task type: This is a classification problem. The classes are unbalanced (33% Low, 42% Medium, and 25% High) in terms of student burnout risk level.

Models compared: Five models were tested on the dataset (Logistic Regression, Decision Tree, KNN, Naive Bayes, and Support Vector Machine).

Evaluation: Accuracy, AUC (one-vs-rest), Precision, Recall, F1-score, and Matthews Correlation Coefficient.

Deliverable: The final product will be a Streamlit app that loads the saved models, takes in a test CSV, and then outputs the metrics and a confusion matrix.
