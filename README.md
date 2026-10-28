# Handling Imbalanced Dataset & Model Optimization

## 📌 Project Overview

This project explores machine learning classification using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. It focuses on understanding class distribution, applying feature scaling techniques, handling class imbalance using class weights, and evaluating a Decision Tree classifier.

The goal is to build and evaluate a classification model that distinguishes between malignant and benign breast cancer cases.

## 🎯 Objectives

- Explore and understand the dataset.
- Perform data inspection and check for missing values.
- Analyze class distribution in training and testing datasets.
- Apply feature scaling techniques.
- Handle class imbalance using class weights.
- Train a Decision Tree classifier.
- Evaluate model performance using classification metrics.

## 📊 Dataset Information

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Source:** Scikit-learn built-in dataset (`sklearn.datasets.load_breast_cancer`)

Dataset details:

- **Total samples:** 569
- **Input features:** 30 numerical features
- **Target variable:** Diagnosis class
- **Missing values:** None detected

The dataset contains features computed from digitized images of breast mass samples, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

### Target Classes

- `0` — Malignant
- `1` — Benign

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook
- Matplotlib

## ⚙️ Project Workflow

### 1. Data Loading and Exploration

The Breast Cancer dataset is loaded using Scikit-learn and converted into a Pandas DataFrame.

Exploratory data analysis includes:

- Inspecting the dataset structure and shape.
- Examining feature names and data types.
- Checking for missing values.
- Exploring descriptive statistics.
- Understanding the distribution of target classes.

### 2. Feature Scaling

Three feature scaling techniques are explored:

- **StandardScaler:** Standardizes features using their mean and standard deviation.
- **MinMaxScaler:** Scales features to a specified range, typically 0 to 1.
- **RobustScaler:** Scales features using the median and interquartile range, making it less sensitive to outliers.

The model training workflow uses features transformed with `RobustScaler`.

### 3. Train-Test Split

The dataset is divided into training and testing sets using an 80:20 split.

- **Training samples:** 455
- **Testing samples:** 114
- **Random state:** 1234
- **Stratification:** Enabled to preserve class proportions.

### 4. Handling Class Imbalance

A class-weighted Decision Tree classifier is used to assign different weights to the target classes.

The model configuration includes:

- `criterion="gini"`
- `max_depth=3`
- `class_weight={0: 2, 1: 1}`

This configuration assigns a higher weight to the malignant class (`0`) to increase its importance during model training.

### 5. Model Training

A Decision Tree classifier is trained on the RobustScaler-transformed training data. Predictions are generated for both the training and testing sets.

### 6. Model Evaluation

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics help assess overall classification performance and the model's ability to identify each target class.

## 📈 Model Evaluation Results

The following results are taken from the notebook's classification report for the **test dataset**.

| Metric | Malignant (0) | Benign (1) |
|---|---:|---:|
| Precision | 0.95 | 0.96 |
| Recall | 0.93 | 0.97 |
| F1-score | 0.94 | 0.97 |
| Support | 42 | 72 |

**Overall test accuracy: 96%**

### Test Confusion Matrix

```text
[[39  3]
 [ 2 70]]
```

Interpretation:

- **39** malignant cases were correctly classified.
- **3** malignant cases were classified as benign.
- **2** benign cases were classified as malignant.
- **70** benign cases were correctly classified.

The model correctly classified 109 out of 114 test samples.

## 🔍 Key Findings

- The dataset contains 569 samples and 30 numerical predictive features.
- No missing values were detected during data inspection.
- RobustScaler was used in the model training workflow.
- Class weights were applied to give greater importance to malignant cases.
- The Decision Tree classifier achieved approximately 96% test accuracy.
- The test classification report shows a malignant-class recall of approximately 93%.

These results describe the performance of the configured Decision Tree on the selected test split. They do not establish how the model would perform on every unseen dataset.

## 🚀 How to Run the Project

1. Clone this repository:

   ```bash
   git clone YOUR_REPOSITORY_URL
   ```

2. Navigate to the project directory:

   ```bash
   cd Handling-Imbalanced-Dataset-Model-Optimization
   ```

3. Install the required libraries:

   ```bash
   pip install pandas scikit-learn matplotlib jupyter
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open the project `.ipynb` file and run the notebook cells.

## 📁 Project Structure

```text
Handling-Imbalanced-Dataset-Model-Optimization/
│
├── README.md
├── LICENSE
├── .gitignore
└── Handling Imbalance Dataset & Model Optimization.ipynb
```

*Note: The notebook filename shown above is illustrative. Use the exact filename uploaded to your repository.*

## 🔮 Future Improvements

- Compare Decision Tree performance with Logistic Regression, Random Forest, and other classifiers.
- Evaluate class imbalance using additional techniques such as SMOTE.
- Compare scaling strategies using consistent train-test splits and evaluation procedures.
- Perform cross-validation and hyperparameter tuning.
- Analyze precision-recall trade-offs and additional metrics relevant to malignant-case detection.

## 👨‍💻 Author

Created as a machine learning project focused on classification, feature scaling, class imbalance, and model evaluation.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
