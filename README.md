# Weather Prediction Project

This project focuses on predicting weather conditions using various machine learning models. The goal is to analyze historical weather data, perform feature engineering, and evaluate different classification algorithms to find the best performing model for predicting daily weather summaries.

## Project Structure

The project is implemented as a Google Colab notebook, containing the following key sections:

1.  **Setup and Data Loading:**
    *   Installation of necessary libraries (pycaret, numpy, pandas, seaborn, matplotlib).
    *   Mounting Google Drive to access the dataset.
    *   Loading and initial exploration of the `weatherHistory.csv` dataset.

2.  **Data Cleaning and Preprocessing:**
    *   Handling missing values (dropping rows with nulls).
    *   Removing duplicate rows.
    *   Analyzing the distribution of categorical features (`Summary`).

3.  **Exploratory Data Analysis (EDA):**
    *   Visualizing the distribution of numerical features using histograms and box plots.
    *   Analyzing correlations between numerical features using a heatmap.
    *   Creating scatter plots to understand relationships between key features (e.g., Temperature vs Apparent Temperature).
    *   Visualizing trends over time (Temperature).
    *   Analyzing the relationship between categorical features and numerical features (Average Temperature by Summary).
    *   Generating pair plots for a subset of numerical features.
    *   Visualizing the distribution of 'Precip Type' (if suitable).

4.  **Feature Engineering:**
    *   Extracting time-based features from the 'Formatted Date' column (Year, DayOfWeek, Is\_Weekend, Hour, Month).
    *   Applying sinusoidal transformations to cyclical features (Hour, Month).
    *   Creating a 'Humidity\_Level' categorical feature.
    *   Generating interaction features between different weather parameters.
    *   Dropping the original 'Formatted Date' column.

5.  **Outlier Handling:**
    *   Visualizing outliers in numerical features using box plots.
    *   Implementing a function to remove outliers using the Interquartile Range (IQR) method.
    *   Applying the outlier removal function to numerical columns.

6.  **Encoding Categorical Features:**
    *   Handling missing values in 'Precip Type'.
    *   Encoding categorical features ('Summary', 'Precip Type', 'Daily Summary') using Label Encoding.
    *   Dropping the original categorical columns.

7.  **Model Training and Evaluation (Manual Approach):**
    *   Splitting the data into training and testing sets.
    *   Scaling numerical features using StandardScaler.
    *   Defining an `evaluate_model` function to calculate and print various classification metrics (Accuracy, Recall, Precision, F1 Score, Classification Report).
    *   Training and evaluating several classification models using pipelines:
        *   Logistic Regression
        *   Decision Tree Classifier
        *   K-Neighbors Classifier
        *   Random Forest Classifier
        *   Support Vector Classifier (SVC)
        *   Gaussian Naive Bayes
    *   Storing and visualizing the performance metrics of these models.

8.  **Model Training and Evaluation (PyCaret):**
    *   Checking and handling classes with only one sample in the target variable, as required by PyCaret.
    *   Setting up the PyCaret environment for classification.
    *   Comparing multiple classification models using `compare_models()`.
    *   Evaluating the best performing model identified by PyCaret.

## Dataset

The dataset used in this project is `weatherHistory.csv`. It contains historical weather data with various attributes such as temperature, humidity, wind speed, pressure, and a daily summary.

## Dependencies

The following libraries are required to run the notebook:

*   `pycaret` (with full, mlops, and time\_series extras)
*   `numpy`
*   `pandas`
*   `seaborn`
*   `matplotlib`
*   `scikit-learn`
*   `scipy`
*   `IPython`

You can install these using `pip` as shown in the notebook.

## How to Run

1.  Open the Google Colab notebook.
2.  Ensure you have the `weatherHistory.csv` dataset available in your Google Drive. Update the path in the notebook if necessary.
3.  Run each code cell sequentially.

## Future Work

*   Hyperparameter tuning for the top performing models identified by PyCaret.
*   Exploring other feature engineering techniques.
*   Investigating different data balancing techniques if class imbalance is a significant issue.
*   Deploying the best performing model.
