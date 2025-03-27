# Wine Quality Prediction Project

## Overview
This project uses machine learning to predict the quality of red wine based on chemical properties like acidity, alcohol content, and pH. The goal is to classify wines into three categories: **Low**, **Medium**, and **High** quality, using the XGBoost algorithm.

- **Dataset**: Red Wine Quality from Kaggle (1,599 samples)
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- **Tools**: Python, pandas, scikit-learn, xgboost

## Project Goals
- Predict wine quality (Low: ≤4, Medium: 5-6, High: ≥7) using 11 chemical features.
- Achieve high classification accuracy with XGBoost and optimize it further.

## Steps
1. **Data Loading**: Loaded `winequality-red.csv` using pandas.
2. **Preprocessing**: Simplified quality scores into 3 categories (Low, Medium, High) and mapped them to numbers (0, 1, 2).
3. **Data Splitting**: Split into 80% training (1,279 samples) and 20% testing (320 samples) sets.
4. **Model Training**: Trained an XGBoost classifier, achieving an initial accuracy of **86.25%**.
5. **Hyperparameter Tuning**: Used GridSearchCV to optimize, resulting in a best accuracy of **85.94%**.

## Results
- **Initial Accuracy**: 86.25% (default XGBoost settings)
- **Tuned Accuracy**: 85.94% (best parameters from GridSearchCV)
- **Best Parameters**: (See `Wine_Quality_Prediction.ipynb` for details—e.g., `max_depth`, `learning_rate`, `n_estimators`)

## Files
- `winequality-red.csv`: The raw dataset from Kaggle.
- `Wine_Quality_Prediction.ipynb`: Jupyter notebook with all code (data loading, preprocessing, modeling, evaluation).
- `xgboost_wine_model.pkl`: Saved tuned XGBoost model.
- `README.md`: This file.

## How to Run
1. **Prerequisites**:
   - Python 3.x
   - Install dependencies: `pip install pandas scikit-learn xgboost joblib`
2. **Steps**:
   - Clone this repository: `git clone https://github.com/JaspreetSingh1024/Wine_Quality_Project.git`
   - Open `Wine_Quality_Prediction.ipynb` in Jupyter Notebook or VS Code.
   - Run all cells to reproduce the results.
3. **Load the Model**:
   - Use `joblib.load('xgboost_wine_model.pkl')` to load the trained model for predictions.

## Future Improvements
- Handle class imbalance (mostly Medium quality) with oversampling or weights.
- Test additional features or regression for exact scores.
- Experiment with other models (e.g., Random Forest, Neural Networks).

## Author
- **Jaspreet Singh**
- LinkedIn: [jaspreet-singh-itsupport](https://www.linkedin.com/in/jaspreet-singh-itsupport/)
- Email: jaspreetsingh1024@outlook.com
