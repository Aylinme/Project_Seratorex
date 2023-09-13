# Project_Seratorex
Objective: The aim is to create a decision tree model for predicting the appropriate drug for future patients with the same medical condition
The dataset consists of features such as Sex (gender), BP (blood pressure), sodium-potassium balance (Na_to_K), and cholesterol (Cholesterol), with the target variable being the drug to which each patient responds (Drug A, Drug B, Drug C, Drug X, or Drug Y).

Steps followed to create the decision tree model:

Data Preprocessing: This part involves handling missing values, encoding categorical variables (Sex, BP, Cholesterol), and normalizing numerical features if necessary.

Data Splitting: The dataset has been divided into two parts, namely the training set and the test set. The training set is used to build the decision tree model, while the test set is used to evaluate its performance.

Building the Decision Tree: The decision tree model is constructed using the training set. The decision tree algorithm will automatically select the most informative features and decision points to make predictions.

Model Evaluation: The test set is used to assess the performance of the decision tree model. Accuracy, recall, and F1 scores are calculated to evaluate how well the model predicts drug responses.

Predicting the Drug for New Patients: After training and evaluating the decision tree model, it can be used to predict the appropriate drug for a new patient with the same medical condition. When the patient's features (age, gender, blood pressure, sodium-potassium, and cholesterol) are input into the decision tree, the predicted drug will be determined.
