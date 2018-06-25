# capstone-proposal
# Using Supervised Learning to predict whether an employee will stay in company or not
# Project: Capstone Project
Install
This project requires Python 3.x and the following Python libraries installed:

NumPy
Pandas
matplotlib
scikit-learn

You will also need to have software installed to run and execute an iPython Notebook

We recommend to install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

#Code
Code is provided in the capstone-project-code.ipynb notebook file. It uses the HR_comma_sep.csv dataset file. Code uses numpy, python, matplotlib and scikit learn libraries. Code uses classification technique of Supervised Learning to train the model. Code has implemented ADABoost Classifier, Gradient Descent Classifier, SVM and Stochastic gradient Descent to calculate performance based on the different methods of training using different techniques.

#Data
This data set contains total of 14999 rows and 10 columns. Target variable (left) is imbalance dataset. It contains 3751 records of employee who have left the company. 11428 records of employee who stayed in the company. Here we are trying to predict employee who can quit. Dataset has been extracted from Kaggle.

Dataset Link: https://www.kaggle.com/giripujar/hr-analytics/data#HR_comma_sep.csv 

# Features
satisfaction_level : Level of Satisfaction
last_evaluation :Time since Last performance Evaluation
number_project :Number of Project completed while at work
average_montly_hours : Average monthly hours at workplace
time_spend_company : Number of years spent in the company
Work_accident : Whether the employee had a workplace accident
promotion_last_5years: Wehter employee was promoted in last 5 years
sales : Department they work for
Salary : Relative level of Salary(high)
# Target Variable

left : Whether employee left the workplace or not (0 - stayed and 1 - Left)

#References

•	https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
•	https://machinelearningmastery.com/logistic-regression-for-machine-learning/
•	https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
•	https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f
•	http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
•	http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
•	https://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/
•	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
•	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
•	http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
•	http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
•	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
