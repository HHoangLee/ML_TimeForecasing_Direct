# ML_TimeForecasing_Direct
This project predicts future CO₂ levels based on historical data using a multi-step forecasting approach.  
We train multiple Linear Regression models — each predicting one step ahead (1st model predicts t+1, 2nd predicts t+2, ...).

## Tech Stack
- Python 3.x
- Pandas, NumPy
- Scikit-Learn
- Matplotlib (optional for visualization)

## Installation
git clone https://github.com/<your-username>/Time-Series-Forecasting.git
cd Time-Series-Forecasting
pip install -r requirements.txt

## Usage
Run the training script:
python src/train_model.py

## This will:
Load CO₂ dataset from data/co2.csv
Generate features & targets using a sliding window
Train 5 regression models for multi-step prediction
Print MAE, MSE, RMSE, and R² for each model

## Notes: 
Replace data/co2.csv with your own time series data (must have time and co2 columns).
You can increase window_size or target_size in train_model.py to experiment with different forecasting horizons.

## Author: 
Maintained by Lê Huy Hoàng
