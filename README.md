# 🏥 Medical Insurance Cost Prediction

This project builds a comprehensive R-based pipeline to **analyze**, **predict**, and **visualize** medical insurance costs using regression models, machine learning techniques (Random Forest, Lasso, Ridge), and an interactive **Shiny Web App**.

---

## 📌 Features

- Exploratory Data Analysis (EDA) with visualizations
- Multiple Linear Regression
- Regularized Regression: Ridge & Lasso
- Random Forest Regression
- Model Evaluation (RMSE, R², Residuals, etc.)
- Shiny App for user interaction and prediction deployment

---

## 📁 Dataset

- **Source:** `insurance.csv` (not included in this repo — use the public ["Medical Cost Personal Datasets"](https://www.kaggle.com/datasets/mirichoi0218/insurance) from Kaggle, or any dataset with the same columns, and place it in the project directory)
- **Columns:**
  - `age`: Age of primary beneficiary
  - `sex`: Gender
  - `bmi`: Body Mass Index
  - `children`: Number of children covered by insurance
  - `smoker`: Smoker status (`yes`/`no`)
  - `region`: Residential region (`southwest`, `southeast`, `northwest`, `northeast`)
  - `charges`: Individual medical costs billed by health insurance

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/bharat3645/Medical-Insurance-Cost-Prediction-.git
cd Medical-Insurance-Cost-Prediction-
```

2. Install the required R packages:

```r
install.packages(c("ggplot2", "GGally", "caret", "MASS", "glmnet",
                    "corrplot", "dplyr", "lmtest", "randomForest", "shiny"))
```

3. Add `insurance.csv` to the project directory (see [Dataset](#-dataset) above), then run the script:

```r
source("Medical-Insaurnce-Cost-Prediction_ver2.R")
```

This runs the EDA and regression / Random Forest models, prints evaluation metrics (RMSE, R²), and launches the Shiny app for interactive cost prediction.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
