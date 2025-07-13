# 🏡 Boston Housing Price Prediction
This project uses the classic **Boston Housing dataset** to build a **Multiple Linear Regression** model for predicting house prices. It includes **exploratory data analysis (EDA)**, **correlation analysis**, **model training**, and **evaluation** using key performance metrics.

---

## 📌 Project Objectives

- Understand the relationship between housing features and median house price (`MEDV`)
- Visualize feature correlations using heatmaps and scatter plots
- Build a Multiple Linear Regression model
- Evaluate model performance using MAE and R² Score
- Visualize predictions against actual values

---

## 📊 Dataset Information

The **Boston Housing dataset** contains information about various housing attributes in Boston suburbs.

| Feature | Description |
|---------|-------------|
| CRIM    | Per capita crime rate by town |
| ZN      | Proportion of residential land zoned for large lots |
| INDUS   | Proportion of non-retail business acres per town |
| CHAS    | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| NOX     | Nitric oxides concentration (parts per 10 million) |
| RM      | Average number of rooms per dwelling |
| AGE     | Proportion of owner-occupied units built before 1940 |
| DIS     | Weighted distances to employment centers |
| RAD     | Index of accessibility to radial highways |
| TAX     | Property tax rate |
| PTRATIO | Pupil-teacher ratio by town |
| LSTAT   | % lower status of the population |
| MEDV    | **Target variable**: Median house value in $1000s |

---

## 🔍 Workflow Overview

1. **Data Import & Cleaning**
   - Loaded dataset using `pd.read_csv("boston.csv")`
   - DataFrame created directly from CSV and verified using `.info()` and `.describe()`

2. **Exploratory Data Analysis (EDA)**
   - Descriptive statistics and structure
   - Heatmap of feature correlations
   - Focus on key features like `RM`, `LSTAT`, `TAX`, `NOX`, etc.
3. **Visualization**
   - Scatter plots between features and target (`MEDV`)
   - Actual vs Predicted comparison plot
4. **Model Building**
   - Used `LinearRegression()` from scikit-learn
   - Applied `train_test_split()` to separate training and test data
5. **Model Evaluation**
   - Calculated **Mean Absolute Error (MAE)**
   - Visualization of model performance

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | ~3.20 |

🔹 The model performs reasonably well. Most predictions are close to the actual values with an average error of ~$3,200.

---

## 📉 Visual Output Examples

### 🔹 Correlation Heatmap  
- Highlights strong positive & negative correlations.
- `RM` (positive) and `LSTAT` (negative) were most influential.

### 🔹 Actual vs Predicted Plot  
- Red dots = model predictions  
- Black diagonal = perfect prediction line  
- Most points lie close to the diagonal → good model fit

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| Pandas, NumPy | Data handling and processing |
| Matplotlib, Seaborn | Data visualization |
| Scikit-learn | Machine learning (model, metrics) |

---
## 🚀 Future Improvements

- Add **Ridge & Lasso Regression** to reduce overfitting from multicollinearity
- Perform **hyperparameter tuning**
- Deploy the model with **Streamlit or Flask**

**Pooja Choudhary**   
💻 BCA (Data Science) | Aspiring Machine Learning Engineer

## ⭐ If you liked this project...

- Leave a ⭐ on the repo

