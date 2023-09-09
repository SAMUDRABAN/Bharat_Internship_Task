## Description

# Imports and Data Loading:

Required libraries, such as pandas, sklearn, and matplotlib, are imported.
A dataset named "melbourne_housing.csv" is read into a DataFrame df.
Basic Data Exploration:

The top rows of the dataset are printed using the head() method.
# Data Preprocessing:

The target variable 'Price' is separated from the features, resulting in X (features) and y (target).
The dataset is split into training (X_train and y_train) and testing (X_test and y_test) sets.
Only numeric features are selected from X_train and X_test.
Any rows containing NaN values in X_train and corresponding rows in y_train are dropped. Similarly, any rows with NaN values in X_test and corresponding rows in y_test are dropped.
Model Building and Prediction:

A linear regression model is instantiated and trained on X_train and y_train.
Predictions are made on the X_test dataset.
Model Evaluation:

The mean squared error between the actual and predicted values is computed and printed.
# Data Visualization:

A heatmap is created to show the correlation between numeric features in X_test.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/4a558aa4-9446-450f-823a-29da83f1ea5a)

Histograms are plotted for each numeric feature in X_test.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/b00bf895-08ef-4f89-bd47-872e8c27dddb)

Boxplots are generated for each numeric feature in X_test to visualize outliers and the spread of data.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/dc151102-be84-42f2-826c-0f0220b1e888)

Scatter plots are generated to visualize the relationship between each feature in X_test and the target variable y_test.
3D Visualization:

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/80955d22-8f24-4811-b104-abf20068554c)

Scatter plots in 3D space are created for actual and predicted values, using the first two numeric features of X_test for the x and y axes, and the target values/predictions for the z-axis.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/1b8c148d-e40d-4d62-ac0a-913a147e6cbb)

Another 3D scatter plot is shown, juxtaposing actual and predicted values in the same plot space.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/9a8373e2-4481-4b8f-a0a8-c2d3e5d99fe0)

A 3D line plot is created to visualize the predicted values.

![image](https://github.com/swetanshusbp/Bharat_Intern_Task/assets/84852778/11a908c7-0cf6-4cb2-9f58-8badfe32074f)

# This code aims to provide an end-to-end demonstration of a linear regression modeling task – from loading the data, preprocessing, modeling, to visualizing results.
