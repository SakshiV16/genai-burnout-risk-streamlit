


Objective: To predict a student’s burnout risk (Low, Medium, or High) from academic, AI usage, study behaviour, and institutional factors.
Context: Generative AI is now in widespread use by students, and heavy use may be linked to wellbeing outcomes such as burnout. This project is looking at the data for patterns and does not claim that use of AI leads to burnout.
Data: The “Impact of AI on Students” dataset from Kaggle (https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students), accessed: 28/10/2026, containing 50,000 student records. After removal of the student ID, there are [14] input features remaining.
Task type: Multi-class classification with three classes of moderately imbalanced size (approx. 33% Low, 42% Medium, and 25% High).
Models compared: Logistic Regression, Decision Tree, K-Nearest Neighbours, Naive Bayes(Gaussian or Multinomial), and Support Vector Machine.
Evaluation: Accuracy, AUC (one-vs-rest), Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).
Deliverable: A Streamlit app that will load the saved models (no retraining required), take a test CSV as input, allow the user to select a model, and then output the metrics and the confusion matrix.
