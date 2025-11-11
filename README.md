
# Regression Project

### Overview
This project explores **Regression Analysis** using classical machine learning algorithms to predict the **temperature** of **Pasig City** based on Philippine weather data from **2020 to March 2025**.

The goal is to identify the **best regression model** for forecasting future temperature trends — specifically from **April to December 2025**.

---

## 1. What is Regression?

Regression is a **supervised machine learning technique** used to predict **continuous numeric values**.  
It learns the relationship between one or more **independent variables (features)** and a **dependent variable (target)**.

Mathematically:
\[
y = f(X) + \epsilon
\]
Where:
- **X** → Input features (e.g., mean temperature, rainfall, wind speed)  
- **y** → Output variable (apparent temperature max)  
- **f(X)** → Function learned by the model  
- **ε** → Error term

---

## 2. How Regression Works

1. **Collect and Prepare Data**  
   - Use labeled data where the target variable is known.  
   - Example: Weather records with measured temperature.

2. **Train a Model**  
   - The model learns patterns mapping inputs → output.

3. **Evaluate Performance**  
   - Check how well predictions match actual values using metrics like MSE, MAE, and R².

4. **Predict New Values**  
   - Use the trained model to forecast unseen or future data.

---

## 3. Choosing the Best Regression Model

Choosing a regression algorithm depends on **data patterns**, **relationships**, and **noise level**.

| Data Pattern | Recommended Model | Notes |
|---------------|------------------|--------|
| Linear relationship | **Linear Regression** | Simple, interpretable baseline |
| Slightly curved trend | **Polynomial Regression** | Fits quadratic or higher-order patterns |
| Complex nonlinear pattern | **Support Vector Regressor (SVR)** | Smooth nonlinear mapping |
| Localized or clustered data | **KNN Regressor** | Works well with neighborhood similarity |
| Hierarchical / rule-based data | **Decision Tree** | Easy to interpret, captures thresholds |
| High variance or noise | **Random Forest** | Robust, ensemble of many trees |
| Multicollinearity present | **Ridge / Lasso Regression** | Regularization to prevent overfitting |

**Best practice:**  
- Start with a **simple model (Linear Regression)** as a baseline.  
- Evaluate others based on **performance metrics** and **data complexity**.  
- Use **cross-validation** to ensure model generalization.

---

## 4. Evaluation Metrics

| Metric | Formula | Description | Goal |
|---------|----------|-------------|------|
| **MAE (Mean Absolute Error)** | \( \frac{1}{n}\sum |y_i - \hat{y}_i| \) | Average absolute deviation | Lower is better |
| **MSE (Mean Squared Error)** | \( \frac{1}{n}\sum (y_i - \hat{y}_i)^2 \) | Penalizes large errors | Lower is better |
| **RMSE (Root MSE)** | \( \sqrt{MSE} \) | Error in same unit as target | Lower is better |
| **R² Score (Coefficient of Determination)** | \( 1 - \frac{SS_{res}}{SS_{tot}} \) | Measures model fit quality | Closer to 1 is better |

---

## 5. Regression Models for Comparison

| Model | Library | Description |
|--------|----------|-------------|
| **Linear Regression** | `sklearn.linear_model.LinearRegression` | Fits a straight line; simple and interpretable |
| **Polynomial Regression** | `PolynomialFeatures + LinearRegression` | Models curves and trends |
| **SVR (Support Vector Regressor)** | `sklearn.svm.SVR` | Finds a margin-based best fit curve |
| **KNN Regressor** | `sklearn.neighbors.KNeighborsRegressor` | Predicts by averaging nearby data points |
| **Decision Tree Regressor** | `sklearn.tree.DecisionTreeRegressor` | Splits data based on thresholds |
| **Random Forest Regressor** | `sklearn.ensemble.RandomForestRegressor` | Combines multiple trees for better accuracy |

---

## 6. Model Evaluation and Comparison

Each model will be evaluated using:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

Results will be visualized using **bar charts** for side-by-side comparison:

- 🔹 Bar chart 1 → MSE comparison  
- 🔹 Bar chart 2 → MAE comparison  
- 🔹 Bar chart 3 → R² comparison  

The **best model** will be selected based on **lowest MSE/MAE** and **highest R²**.

---

## 7. Mathematical Foundation

### Linear Regression Equation
\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon
\]
**Goal:** minimize Mean Squared Error (MSE).

### Optimization Objective
\[
\text{Minimize: } \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
\]

Where:
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( n \) = number of samples  

---

## 8. Dataset Information

| Column | Description |
|---------|-------------|
| city_name | Name of city |
| datetime | Date of weather record |
| weather_code | Weather condition identifier |
| temperature_2m_max / min / mean | Observed air temperature (°C) |
| apparent_temperature_max / min / mean | Feels-like temperature (°C) — target variable |
| precipitation_sum, rain_sum, snowfall_sum | Total rain/snow accumulation |
| precipitation_hours | Hours of precipitation during the day |
| wind_speed_10m_max, wind_gusts_10m_max | Maximum daily wind speed and gusts |
| wind_direction_10m_dominant | Dominant wind direction (°) |
| sunshine_duration | Total daily sunshine hours |
| daylight_duration | Total daylight hours |
| shortwave_radiation_sum | Daily solar radiation |
| et0_fao_evapotranspiration | Evapotranspiration index (°C equivalent) |

**Filtered subset used for modeling:**
- City: `Pasig`
- Date range: `2020-01-01` → `2025-03-31`
- Target variable: `apparent_temperature_max`
- Input features: selected weather and time-based variables after correlation analysis

---

## 9. Data Cleaning Steps

1. **Filter** only the records where `city_name == 'Pasig'`.  
2. **Convert** `datetime` to a proper datetime format and **sort chronologically**.  
3. **Handle missing or null values** — use interpolation or mean imputation if necessary.  
4. **Extract time features** — `year`, `month`, and `day` from the datetime column to capture seasonal behavior.  
5. **Check data consistency** — ensure no duplicates, unrealistic temperature values, or missing date ranges.  
6. **Prepare features** for modeling (select relevant columns, drop unneeded ones).

---

## 10. Exploratory Data Analysis (EDA)

The EDA stage will provide insights into trends, seasonality, and feature relationships.

### Key EDA Objectives:
- Visualize the **trend of apparent_temperature_max** over time.  
- Analyze **monthly and yearly averages** to capture seasonal variations.  
- Detect **missing data, outliers**, or unusual spikes in weather readings.  
- Create a **correlation heatmap** to identify which features are strongly related to the target variable.  
- Use a **feature correlation bar chart** to rank the most influential predictors.  
- Select **strong and non-redundant features** for model training.

### Correlation Analysis
Correlation will help identify the most relevant predictors.  
Features with a **high positive correlation** (close to +1) or **high negative correlation** (close to -1) with `apparent_temperature_max` are good candidates.

Example features expected to correlate:
- Positive: `temperature_2m_mean`, `sunshine_duration`, `shortwave_radiation_sum`
- Negative: `precipitation_sum`, `wind_speed_10m_max`

---

## 11. Workflow Summary

1. **Load and clean the dataset** (filter Pasig, convert datetime, handle missing data).  
2. **Conduct EDA**: visualize trends, correlations, and select strong features.  
3. **Split the dataset** into Train/Test subsets (e.g., 80/20 ratio).  
4. **Apply feature scaling** for models sensitive to magnitude (SVR, KNN, Polynomial).  
5. **Train multiple regression models** (Linear, Polynomial, SVR, KNN, Decision Tree, Random Forest).  
6. **Evaluate** each model using MSE, MAE, and R² metrics.  
7. **Visualize model comparison** with bar charts for MSE, MAE, and R².  
8. **Select the best-performing model** based on lowest error and highest R² score.  
9. **Forecast April–December 2025** apparent maximum temperature for Pasig City.  
10. **Plot actual vs predicted values** to visualize forecast accuracy.
---

## 12. Model Evaluation Visualization

The notebook will generate the following:
- **MSE Comparison Bar Chart**  
- **MAE Comparison Bar Chart**  
- **R² Comparison Bar Chart**

These visualizations will clearly identify which regression model performs best on the Pasig dataset.

---

## 13. Forecasting (Next Step)

Once the best model is identified:
- Retrain it using the **entire dataset (2020–March 2025)**  
- Forecast **Pasig City’s apparent_temperature_max** from **April–December 2025**
- Plot predictions vs. actual trend for visualization

---

## 14. Tools and Libraries

| Tool | Purpose |
|------|----------|
| **Python (3.10+)** | Main programming language |
| **JupyterLab** | Interactive notebook for analysis |
| **pandas, numpy** | Data manipulation |
| **matplotlib, seaborn** | Data visualization |
| **scikit-learn** | Regression modeling and evaluation |

---

## 15. Author

**Johnny Pillazo**  
Master’s in Data Science  
Philippines

---

> _“Regression models help us uncover continuous relationships in data — enabling informed decisions, forecasting, and deeper understanding of patterns hidden in numbers.”_


## Environment Setup

Before running this project, set up a clean Python virtual environment.

### 1. Create and activate the virtual environment
```bash
python -m venv venv
```
```bash
venv\Scripts\activate
```
Reminder: Always make sure you see (venv) in your terminal before installing or running anything!
Example:

```bash
(venv) PS C:\Users\[name]\Desktop\Regression>
```

### 2. Install required libraries
```bash
pip install -r requirements.txt
```

or, for the first setup:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab
```
```bash
pip freeze > requirements.txt
```

### 3. Launch JupyterLab
```bash
jupyter lab
```