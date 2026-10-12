# Benchmark summary

Times are mean ± standard deviation over 9 measured runs (after warm-up), in milliseconds.
Speedup = BareBones mean / implementation mean (> 1 is faster). Accuracy and macro F1 are on the held-out test split.

## Speedup factors vs BareBones (end-to-end: transfer + fit + predict)

| Dataset | Model | Intelex | CuPy | Δ macro F1 Intelex | Δ macro F1 CuPy |
|---|---|---|---|---|---|
| active_Wiretap | Random Forest Classifier | – | – | +nan | +nan |
| active_Wiretap | Logistic Regression | – | – | +nan | +nan |
| active_Wiretap | Extra Trees Classifier | – | – | +nan | +nan |
| cyberbully detection | Random Forest Classifier | – | – | +nan | +nan |
| cyberbully detection | Logistic Regression | – | – | +nan | +nan |
| cyberbully detection | Extra Trees Classifier | – | – | +nan | +nan |
| diabetes | Random Forest Classifier | – | – | +nan | +nan |
| diabetes | Logistic Regression | – | – | +nan | +nan |
| diabetes | Extra Trees Classifier | – | – | +nan | +nan |
| loan prediction | Random Forest Classifier | – | – | +nan | +nan |
| loan prediction | Logistic Regression | – | – | +nan | +nan |
| loan prediction | Extra Trees Classifier | – | – | +nan | +nan |
| student_performance | Random Forest Classifier | – | – | +nan | +nan |
| student_performance | Logistic Regression | – | – | +nan | +nan |
| student_performance | Extra Trees Classifier | – | – | +nan | +nan |

## active_Wiretap

| Model | Impl | Fit ms | Predict ms | Transfer ms | Total ms | Speedup (fit) | Speedup (total) | Accuracy | Macro F1 | Backend |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest Classifier | Intelex | 20,838 ± 216 | 607 ± 29 | 0.0 ± 0.0 | 21,444 ± 197 | – | – | 1.0000 | 1.0000 | `sklearnex.ensemble._forest` |
| Random Forest Classifier | CuPY | 79,710 ± 264 | 514 ± 20 | 311 ± 39 | 80,534 ± 271 | – | – | 1.0000 | 1.0000 | `sklearn.ensemble._forest` |
| Logistic Regression | Intelex | 666 ± 6 | 20.2 ± 1.9 | 0.0 ± 0.0 | 687 ± 5 | – | – | 0.7661 | 0.7579 | `sklearnex.linear_model.logistic_regression` |
| Logistic Regression | CuPY | 4,189 ± 18 | 17.7 ± 0.3 | 341 ± 15 | 4,548 ± 26 | – | – | 0.7675 | 0.7591 | `sklearn.linear_model._logistic` |
| Extra Trees Classifier | Intelex | 39,954 ± 606 | 818 ± 161 | 0.0 ± 0.0 | 40,773 ± 517 | – | – | 1.0000 | 1.0000 | `sklearnex.ensemble._forest` |
| Extra Trees Classifier | CuPY | 48,601 ± 88 | 716 ± 14 | 324 ± 30 | 49,642 ± 101 | – | – | 1.0000 | 1.0000 | `sklearn.ensemble._forest` |

## cyberbully detection

| Model | Impl | Fit ms | Predict ms | Transfer ms | Total ms | Speedup (fit) | Speedup (total) | Accuracy | Macro F1 | Backend |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest Classifier | Intelex | 1,955 ± 9 | 41.8 ± 0.3 | 0.0 ± 0.0 | 1,996 ± 9 | – | – | 0.6420 | 0.6109 | `sklearnex.ensemble._forest` |
| Random Forest Classifier | CuPY | 1,946 ± 17 | 42.6 ± 0.4 | 4.2 ± 1.3 | 1,993 ± 18 | – | – | 0.6420 | 0.6109 | `sklearn.ensemble._forest` |
| Logistic Regression | Intelex | 838 ± 3 | 1.4 ± 0.0 | 0.0 ± 0.0 | 839 ± 3 | – | – | 0.5020 | 0.4945 | `sklearnex.linear_model.logistic_regression` |
| Logistic Regression | CuPY | 838 ± 2 | 0.6 ± 0.0 | 4.8 ± 0.1 | 843 ± 2 | – | – | 0.5020 | 0.4945 | `sklearn.linear_model._logistic` |
| Extra Trees Classifier | Intelex | 2,728 ± 60 | 44.6 ± 4.5 | 0.0 ± 0.0 | 2,772 ± 58 | – | – | 0.6524 | 0.6172 | `sklearnex.ensemble._forest` |
| Extra Trees Classifier | CuPY | 2,717 ± 17 | 47.7 ± 5.4 | 5.0 ± 0.1 | 2,770 ± 18 | – | – | 0.6524 | 0.6172 | `sklearn.ensemble._forest` |

## diabetes

| Model | Impl | Fit ms | Predict ms | Transfer ms | Total ms | Speedup (fit) | Speedup (total) | Accuracy | Macro F1 | Backend |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest Classifier | Intelex | 197 ± 14 | 21.3 ± 2.3 | 0.0 ± 0.0 | 219 ± 15 | – | – | 0.9697 | 0.8894 | `sklearnex.ensemble._forest` |
| Random Forest Classifier | CuPY | 527 ± 10 | 42.2 ± 0.4 | 2.0 ± 0.1 | 572 ± 10 | – | – | 0.9699 | 0.8898 | `sklearn.ensemble._forest` |
| Logistic Regression | Intelex | 9.6 ± 0.2 | 1.3 ± 0.1 | 0.0 ± 0.0 | 10.9 ± 0.2 | – | – | 0.9603 | 0.8550 | `sklearnex.linear_model.logistic_regression` |
| Logistic Regression | CuPY | 33.7 ± 0.3 | 0.4 ± 0.0 | 1.8 ± 0.1 | 35.9 ± 0.3 | – | – | 0.9603 | 0.8550 | `sklearn.linear_model._logistic` |
| Extra Trees Classifier | Intelex | 731 ± 87 | 23.4 ± 0.6 | 0.0 ± 0.0 | 754 ± 87 | – | – | 0.9667 | 0.8819 | `sklearnex.ensemble._forest` |
| Extra Trees Classifier | CuPY | 406 ± 11 | 43.4 ± 4.0 | 5.0 ± 0.7 | 454 ± 12 | – | – | 0.9677 | 0.8835 | `sklearn.ensemble._forest` |

## loan prediction

| Model | Impl | Fit ms | Predict ms | Transfer ms | Total ms | Speedup (fit) | Speedup (total) | Accuracy | Macro F1 | Backend |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest Classifier | Intelex | 44,378 ± 365 | 197 ± 16 | 0.0 ± 0.0 | 44,575 ± 349 | – | – | 0.8976 | 0.7530 | `sklearnex.ensemble._forest` |
| Random Forest Classifier | CuPY | 41,448 ± 120 | 222 ± 14 | 329 ± 36 | 41,999 ± 104 | – | – | 0.8971 | 0.7519 | `sklearn.ensemble._forest` |
| Logistic Regression | Intelex | 897 ± 13 | 15.5 ± 2.2 | 0.0 ± 0.0 | 912 ± 14 | – | – | 0.8770 | 0.4672 | `sklearnex.linear_model.logistic_regression` |
| Logistic Regression | CuPY | 398 ± 10 | 12.6 ± 0.1 | 158 ± 7 | 568 ± 11 | – | – | 0.8770 | 0.4672 | `sklearn.linear_model._logistic` |
| Extra Trees Classifier | Intelex | 64,161 ± 578 | 233 ± 17 | 0.0 ± 0.0 | 64,394 ± 564 | – | – | 0.8970 | 0.7494 | `sklearnex.ensemble._forest` |
| Extra Trees Classifier | CuPY | 78,417 ± 205 | 262 ± 7 | 341 ± 41 | 79,021 ± 214 | – | – | 0.8976 | 0.7499 | `sklearn.ensemble._forest` |

## student_performance

| Model | Impl | Fit ms | Predict ms | Transfer ms | Total ms | Speedup (fit) | Speedup (total) | Accuracy | Macro F1 | Backend |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest Classifier | Intelex | 2.8 ± 0.4 | 1.7 ± 0.0 | 0.0 ± 0.0 | 4.4 ± 0.5 | – | – | 1.0000 | 1.0000 | `sklearnex.ensemble._forest` |
| Random Forest Classifier | CuPY | 128 ± 4 | 29.4 ± 0.5 | 0.4 ± 0.1 | 158 ± 4 | – | – | 1.0000 | 1.0000 | `sklearn.ensemble._forest` |
| Logistic Regression | Intelex | 4.6 ± 0.1 | 0.9 ± 0.1 | 0.0 ± 0.0 | 5.5 ± 0.1 | – | – | 0.9789 | 0.9789 | `sklearnex.linear_model.logistic_regression` |
| Logistic Regression | CuPY | 3.4 ± 0.1 | 0.2 ± 0.0 | 0.3 ± 0.0 | 3.9 ± 0.1 | – | – | 0.9789 | 0.9789 | `sklearn.linear_model._logistic` |
| Extra Trees Classifier | Intelex | 3.1 ± 0.1 | 1.8 ± 0.0 | 0.0 ± 0.0 | 4.9 ± 0.1 | – | – | 0.9859 | 0.9859 | `sklearnex.ensemble._forest` |
| Extra Trees Classifier | CuPY | 97.5 ± 2.2 | 29.5 ± 0.5 | 0.5 ± 0.1 | 128 ± 2 | – | – | 0.9789 | 0.9789 | `sklearn.ensemble._forest` |
