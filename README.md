# Breast Cancer Detection using K-Nearest Neighbors (KNN)

## Project Overview

This project aims to build a model for detecting breast cancer using the K-Nearest Neighbors (KNN) algorithm. The dataset used contains various features extracted from breast mass images. The objective is to classify the tumor as either benign or malignant based on these features.

## Dataset

The dataset contains the following features:

- **id**: ID number of the patient
- **diagnosis**: Diagnosis of breast tissues (M = malignant, B = benign)
- **radius_mean**: Mean of distances from the center to points on the perimeter
- **texture_mean**: Standard deviation of gray-scale values
- **perimeter_mean**: Mean size of the core tumor
- **area_mean**: Mean area of the tumor
- **smoothness_mean**: Mean of local variation in radius lengths
- **compactness_mean**: Mean of perimeter² / area - 1.0
- **concavity_mean**: Mean of the severity of concave portions of the contour
- **concave points_mean**: Mean number of concave portions of the contour
- **symmetry_mean**: Mean symmetry of the tumor
- **fractal_dimension_mean**: Mean "coastline approximation" (1 - fractal dimension)
- **radius_se**: Standard error of distances from the center to points on the perimeter
- **texture_se**: Standard error of gray-scale values
- **perimeter_se**: Standard error of the size of the core tumor
- **area_se**: Standard error of the area of the tumor
- **smoothness_se**: Standard error of local variation in radius lengths
- **compactness_se**: Standard error of perimeter² / area - 1.0
- **concavity_se**: Standard error of the severity of concave portions of the contour
- **concave points_se**: Standard error of the number of concave portions of the contour
- **symmetry_se**: Standard error of symmetry of the tumor
- **fractal_dimension_se**: Standard error of "coastline approximation"
- **radius_worst**: "Worst" or largest mean value for radius
- **texture_worst**: "Worst" or largest mean value for texture
- **perimeter_worst**: "Worst" or largest mean value for perimeter
- **area_worst**: "Worst" or largest mean value for area
- **smoothness_worst**: "Worst" or largest mean value for smoothness
- **compactness_worst**: "Worst" or largest mean value for compactness
- **concavity_worst**: "Worst" or largest mean value for concavity
- **concave points_worst**: "Worst" or largest mean value for concave points
- **symmetry_worst**: "Worst" or largest mean value for symmetry
- **fractal_dimension_worst**: "Worst" or largest mean value for fractal dimension

## Project Notebook

The complete project notebook is available [here](https://github.com/YourUsername/Breast-Cancer-Detection/blob/main/Breast_Cancer_Detection_KNN.ipynb). It includes the following steps:

1. **Data Exploration and Preprocessing**:
    - Load and inspect the dataset.
    - Handle missing values.
    - Encode the 'diagnosis' column (M = 1, B = 0).
    - Normalize the feature values.

2. **Exploratory Data Analysis (EDA)**:
    - Visualize the distribution of features.
    - Analyze the correlation between features and the diagnosis.

3. **Model Building**:
    - Split the data into training and testing sets.
    - Train a K-Nearest Neighbors (KNN) classifier.
    - Perform hyperparameter tuning to find the optimal number of neighbors (k).

4. **Model Evaluation**:
    - Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
    - Compare performance with different values of k.

5. **Prediction and Conclusion**:
    - Make predictions on the test set.
    - Summarize the findings and conclusion.

## Results

The results of the model evaluation and the final predictions on the test set are included in the project notebook. The KNN model achieved high accuracy and provided insights into the most important features for breast cancer detection.

## Conclusion

This project demonstrates the application of the K-Nearest Neighbors algorithm for breast cancer detection. By leveraging data preprocessing, exploratory data analysis, and model evaluation, we can build accurate predictive models and gain valuable insights into the diagnosis of breast cancer.
