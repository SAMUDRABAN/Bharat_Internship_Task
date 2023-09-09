# Description 
This Python script conducts an analysis on the Iris dataset, a popular dataset used for machine learning and statistics demonstrations. The dataset contains features of iris flowers and their respective species.

# Imports:

The script starts by importing the necessary libraries. These include libraries for data manipulation (numpy, pandas), visualization (matplotlib), and machine learning (sklearn).
Loading Data:

The Iris dataset is loaded from sklearn.datasets. The dataset is then converted into a pandas DataFrame for easy manipulation and visualization.
Feature Selection:

Only two features, the sepal length and petal length, are chosen from the dataset for this analysis. The target variable is the species of the iris flower.
Data Splitting:

The dataset is split into training and testing sets using a 70-30 split.
Data Standardization:

The features are standardized using the StandardScaler to ensure they have a mean of 0 and a standard deviation of 1. This is especially important for algorithms, like SVM, that rely on the magnitude and scale of the data.
Model Training:

A Support Vector Machine (SVM) with a linear kernel is trained on the training data. This classifier will learn to separate the different species of iris based on the two selected features.
Prediction & Evaluation:

The trained SVM classifier is then used to make predictions on the test set.
The classifier's performance is evaluated using the classification report (which provides precision, recall, and F1-score) and the confusion matrix.
Visualization:

A 3D scatter plot visualizes the training data. The X and Y axes represent the standardized values of sepal length and petal length, respectively, while the Z-axis represents the species of the iris. Each data point's color corresponds to its species.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/ca4e6a50-7a89-441a-9874-e54b4a45ce8e)

# User Input & Prediction:

The script then prompts the user to enter values for sepal length and petal length.
Using the trained SVM classifier, the species of iris is predicted based on these input values.

# In essence, this script provides a basic demonstration of how to use a Support Vector Machine to classify iris species based on two of their features. It includes steps for data preprocessing, model training, evaluation, visualization, and user interaction.
