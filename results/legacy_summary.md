# Legacy results (PerformanceSpeeds.csv)

Single run per cell, training time only, no warm-up, no accuracy. In these runs the
Intelex patch was applied after the estimators were imported, so the Intelex column is
most likely unpatched scikit-learn. Kept for reference; superseded by results/summary.md.

| Dataset | Model | BareBones ms | Intelex ms | CuPy ms | Intelex speedup | CuPy speedup |
|---|---|---|---|---|---|---|
| active_Wiretap | XGBoost | 88,000 | 86,000 | 90,000 | 1.02× | 0.98× |
| active_Wiretap | Logistic Regression | 113,000 | 116,000 | 111,000 | 0.97× | 1.02× |
| active_Wiretap | Random Forest Classifier | 218,000 | 219,000 | 227,000 | 1.00× | 0.96× |
| active_Wiretap | AdaBoost | 622,000 | 619,000 | 623,000 | 1.00× | 1.00× |
| active_Wiretap | Multi-layer Perceptron classifier | 521,000 | 522,000 | 284,000 | 1.00× | 1.83× |
| active_Wiretap | LightGBM classifier | 1,040 | 1,028 | 1,028 | 1.01× | 1.01× |
| student_performance | XGBoost | 270 | 112 | 159 | 2.41× | 1.70× |
| student_performance | Logistic Regression | 50.0 | 46.0 | 152 | 1.09× | 0.33× |
| student_performance | Random Forest Classifier | 163 | 167 | 275 | 0.98× | 0.59× |
| student_performance | AdaBoost | 29.0 | 25.0 | 135 | 1.16× | 0.21× |
| student_performance | Multi-layer Perceptron classifier | 83.0 | 82.0 | 186 | 1.01× | 0.45× |
| student_performance | LightGBM classifier | 99.0 | 33.0 | 135 | 3.00× | 0.73× |
| diabetes | XGBoost | 271 | 77.0 | 135 | 3.52× | 2.01× |
| diabetes | Logistic Regression | 62.0 | 61.0 | 118 | 1.02× | 0.53× |
| diabetes | Random Forest Classifier | 673 | 665 | 720 | 1.01× | 0.93× |
| diabetes | AdaBoost | 795 | 769 | 888 | 1.03× | 0.90× |
| diabetes | Multi-layer Perceptron classifier | 1,206 | 1,246 | 1,381 | 0.97× | 0.87× |
| diabetes | LightGBM classifier | 144 | 71.0 | 191 | 2.03× | 0.75× |
| loan prediction | XGBoost | 4,244 | 3,969 | 3,572 | 1.07× | 1.19× |
| loan prediction | Logistic Regression | 9,822 | 9,941 | 37,521 | 0.99× | 0.26× |
| loan prediction | Random Forest Classifier | 147,000 | 155,000 | 302,000 | 0.95× | 0.49× |
| loan prediction | AdaBoost | 7,044 | 7,378 | 8,255 | 0.95× | 0.85× |
| loan prediction | Multi-layer Perceptron classifier | 240,599 | 240,876 | 270,123 | 1.00× | 0.89× |
| loan prediction | LightGBM classifier | 2,491 | 2,327 | 2,255 | 1.07× | 1.10× |
| cyberbully detection | XGBoost | 13,300 | 13,283 | 1,371 | 1.00× | 9.70× |
| cyberbully detection | Logistic Regression | 1,157 | 1,179 | 1,357 | 0.98× | 0.85× |
| cyberbully detection | Random Forest Classifier | 22,993 | 23,306 | 23,252 | 0.99× | 0.99× |
| cyberbully detection | AdaBoost | 3,322 | 3,317 | 91,000 | 1.00× | 0.04× |
| cyberbully detection | Multi-layer Perceptron classifier | 10,762 | 11,072 | 8,500 | 0.97× | 1.27× |
| cyberbully detection | LightGBM classifier | 3,560 | 3,253 | 3,988 | 1.09× | 0.89× |
