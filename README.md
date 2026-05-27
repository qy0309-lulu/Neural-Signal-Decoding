# Neural-Signal-Decoding

## Project Overview
This project implements decoding and prediction for neural signal datasets using a variety of classic algorithms, mapping neural signals to behavioral variables.

## Methods
- GRU, LSTM, RNN: Temporal neural signal decoding based on recurrent neural networks
- XGBoost: Regression prediction using gradient boosting trees
- KalmanFilter: Linear decoding with Kalman Filter

## Datasets
1. **cori dataset**: Neural signal decoding task
2. **crcns hc2 dataset**: Position prediction task (predict x-coordinate and y-coordinate separately)

Note: This project only uses data from one session for each dataset.

## File Structure
Each algorithm is organized in a separate folder with corresponding scripts for the two datasets:

```text
├── GRU/
│   ├── cori.py
│   └── hc2.py
├── LSTM/
│   ├── cori.py
│   └── hc2.py
├── RNN/
│   ├── cori.py
│   └── hc2.py
├── XGBoost/
│   ├── cori.py
│   └── hc2.py
├── KalmanFilter/
│   ├── cori.py
│   └── hc2.py
├── main_hc.py
├── main_cori.py
├── .gitignore
├── LICENSE
└── README.md
```

## Environment
- Python 3.12
- TensorFlow / Keras
- XGBoost
- NumPy, Pandas, Scipy

## Usage
Run the .py files in the corresponding folders directly to perform neural decoding with the target model.