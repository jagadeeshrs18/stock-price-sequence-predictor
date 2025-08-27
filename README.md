# 📈 Stock Price Sequence Predictor

This project predicts the **next-day stock price** using an **LSTM (Long Short-Term Memory) deep learning model** trained on multiple stock datasets (`TCS`, `TataSteel`, `TSLA`).  

The model uses the last **60 closing prices** of a stock to predict the next day's closing price.

---

## 🛠 Tech Stack
- **Python 3.12**
- **TensorFlow / Keras**
- **Pandas, NumPy**
- **Matplotlib (for visualization)**
- **Joblib (for scaler saving/loading)**
- **Colab / Jupyter Notebook** (for training & testing)

---

## 📂 Project Structure
```

├── combined\_stock\_model.h5      # Trained LSTM model
├── combined\_scaler.pkl          # Scaler used for normalization
├── tcs.csv                      # TCS stock data
├── tatasteel.csv                 # Tata Steel stock data
├── tsla.csv                     # Tesla stock data
├── stock\_predictor.ipynb        # Colab notebook (training & testing)
└── README.md

```

---

## ⚙️ Training
- Combined datasets of TCS, TataSteel, and TSLA were used.
- Data preprocessing:
  - Converted column names to lowercase
  - Extracted `close` prices
  - Scaled values using `MinMaxScaler`
- Model architecture:
  - **2 LSTM layers**
  - **Dropout layers** to reduce overfitting
  - **Dense layers** for final prediction

---

## 📊 Model Summary
```

## Layer (type)         Output Shape     Param \#

LSTM (50 units)      (None, 60, 50)   10,400
Dropout              (None, 60, 50)   0
LSTM (50 units)      (None, 50)       20,200
Dropout              (None, 50)       0
Dense (25 units)     (None, 25)       1,275
Dense (1 unit)       (None, 1)        26
----------------------------------------

Total params: 31,903

````

---

## 🔮 Prediction Example
```text
TCS next day predicted price:      3422.33
TataSteel next day predicted price: 1097.98
TSLA next day predicted price:      712.20
````

---

## 🚀 How to Run

1. Clone the repo / copy files.
2. Install dependencies:

   ```bash
   pip install tensorflow pandas numpy scikit-learn joblib
   ```
3. Load the model and scaler:

   ```python
   from tensorflow.keras.models import load_model
   import joblib
   import pandas as pd
   import numpy as np

   model = load_model("combined_stock_model.h5")
   scaler = joblib.load("combined_scaler.pkl")
   ```
4. Run predictions on any stock dataset (CSV with `date, open, high, low, close, volume`).

---

## 📌 Next Steps

* ✅ Build backend with Flask/FastAPI to serve predictions.
* ✅ Add frontend (React/Streamlit) to visualize predictions.
* 🚧 Extend to more stocks by updating the dataset.
 
 
 
