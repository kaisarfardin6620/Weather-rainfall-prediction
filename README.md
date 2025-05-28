# Weather Rainfall Prediction and Modeling

This repository contains code for analyzing weather data and building regression models to predict rainfall. The project explores various regression algorithms, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, and K-Nearest Neighbors Regressor, and uses PyCaret for automated machine learning.

## Project Overview

The goal of this project is to predict rainfall based on historical weather data. The notebook covers the following steps:

1.  **Data Loading and Exploration:** Loading the weather dataset, examining its structure, and performing initial descriptive analysis.
2.  **Feature Engineering:** Creating new features from the 'Date' column to capture temporal patterns (Month, Quarter, Days Since, cyclical month features).
3.  **Data Visualization:** Visualizing the data distribution, correlations, and potential outliers.
4.  **Outlier Detection and Handling:** Identifying and optionally removing outliers using the Interquartile Range (IQR) method.
5.  **Polynomial Feature Creation:** Transforming features using polynomial features to capture non-linear relationships.
6.  **Model Training and Evaluation (Manual):** Training and evaluating several regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor, KNeighbors Regressor) using standard scikit-learn pipelines and metrics.
7.  **Hyperparameter Tuning:** Tuning the hyperparameters of the Decision Tree, Random Forest, and K-Nearest Neighbors models using GridSearchCV.
8.  **Model Comparison:** Comparing the performance of the models before and after tuning.
9.  **Automated ML with PyCaret:** Utilizing the PyCaret library to quickly setup the regression environment, compare various models automatically, and evaluate the best performing model.

## Dataset

The dataset used in this project is `weather.csv`. It is expected to be located in the `/content/drive/MyDrive/dataset/` path if you are running this notebook in Google Colab and have your Google Drive mounted.

*Please ensure you have the `weather.csv` file in the specified location or update the file path in the notebook.*

## Dependencies

The following libraries are required to run the notebook:

*   `numpy`
*   `pandas`
*   `seaborn`
*   `matplotlib`
*   `sklearn`
*   `pycaret[full]`
*   `pycaret[mlops]`
*   `pycaret[time-series]`

You can install the required libraries using pip:
Alternatively, the notebook includes `!pip install` commands to install them within the Colab environment.

## How to Run the Code

1.  **Clone the repository:**
2.  2.  **Open the notebook:** Open the `weather_rainfall_prediction.ipynb` (or whatever you name your notebook file) in Google Colab or a Jupyter environment.
3.  **Mount Google Drive (if using Colab):** The notebook includes code to mount your Google Drive to access the dataset.
4.  **Run the cells:** Execute the cells in the notebook sequentially.

## Results

The notebook provides performance metrics (MSE, RMSE, MAE, R2) for the different models. It also includes visualizations of residuals and actual vs. predicted values to help assess model performance. The PyCaret section will provide an automated comparison of many different regression models.

## Future Work

*   Explore other feature engineering techniques.
*   Investigate different outlier handling strategies.
*   Experiment with more advanced regression models.
*   Implement time series forecasting techniques.
*   Deploy the trained model for making predictions.
