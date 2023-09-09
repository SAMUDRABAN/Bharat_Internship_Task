# Description 
This Python script performs an analysis of a wine quality dataset, specifically focusing on red wines. The dataset contains several physicochemical attributes of the wine and its corresponding quality.

# Imports:

Essential libraries and tools like pandas, seaborn, and matplotlib are imported for data manipulation, analysis, and visualization.
The LinearRegression model from sklearn.linear_model is imported to establish a relationship between wine features and wine quality.
Several utility functions and tools are imported for model evaluation, data preprocessing, and dataset splitting.
Data Loading:

The dataset is loaded from a local path into a dataframe named wine.
The first few rows and data summary are displayed to understand the dataset's structure.
Data Visualization:

A series of bar plots are generated, displaying the relationship between different physicochemical attributes of the wine (like acidity, sugar, and alcohol content) and the wine's quality.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/44473c76-d7df-4817-aa23-13089c978e0a)

# Data Preparation:

The dataset is split into features (X) and the target (y), where the target is the wine quality.
The data is then divided into training and testing sets.
The feature data is standardized using the StandardScaler to improve model performance.
Modeling:

A linear regression model is trained using the training data to predict wine quality.
The model's predictions are then evaluated using Mean Squared Error (MSE) and R^2 score, which are commonly used metrics for regression problems.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/6d15dd26-2d55-4807-8fd5-a4ae6e7d8e8a)

# Residual Analysis:

A residuals vs. fitted values plot is produced to visualize the difference between the predicted and actual values.
A histogram displays the distribution of residuals to check for normality and understand prediction errors.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/05930afb-767d-4c5e-a7e4-bd7c4d390f1f)

The importance of each feature in the linear regression model is visualized using a bar plot of the coefficients.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/76ea2346-00cf-490e-99a7-8fc844f168c9)


# Correlation Analysis:

A heatmap is generated to visualize the correlation between different features in the dataset. This gives insights into which features might be redundant or which features heavily influence wine quality.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/7b2abd40-5f48-4283-bd83-3819c80cdf99)

3D Visualization:

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/daaa2fc4-93ab-4cc1-ae98-f0a3a4ffadc5)


Using the plotly library, a 3D scatter plot is created to visualize the relationship between three features ('fixed acidity', 'volatile acidity', and 'citric acid') against wine quality. This provides an intuitive way to understand how these features interact with each other concerning wine quality.
# In summary, this script performs an end-to-end analysis of a dataset describing red wine's quality. The primary objective is to understand which features influence wine quality the most and model this relationship using linear regression. The analysis includes data visualization, model training, evaluation, and residual analysis.
