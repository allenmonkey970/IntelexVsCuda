# IntelexVsCuda

**Benchmarking Intel® Extension for Scikit-learn (scikit-learn-intelex) vs. CuPy-accelerated methods for machine learning workflows.**

## Overview
This project benchmarks and compares the performance of various machine learning algorithms using [Intel's scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) and [NVIDIA's CuPy](https://cupy.dev/) acceleration across multiple datasets. The goal is to determine which optimization—CPU-based (Intelex) or GPU-based (CuPy)—performs better for specific algorithms and dataset types, as part of a college research project.

- **scikit-learn-intelex**: Optimizes select scikit-learn estimators using low-level CPU instructions and multi-threading for faster performance.
- **CuPy**: GPU array library with a NumPy-compatible API; enables GPU acceleration for array computations.

## Setup

**Install the required dependencies:**
```bash
pip install scikit-learn scikit-learn-intelex cupy-cuda xgboost
```
If you encounter missing dependencies, also install:
```bash
pip install pandas matplotlib numpy
```
> _Note: Some datasets or algorithms may require additional libraries (e.g., warnings for suppression)._

## Experiment Methodology

1. **Preprocessing**: Each dataset was cleaned and preprocessed to fit the requirements of the algorithms and libraries.
2. **Model Selection**: The following ML models were tested:
    - LightGBM
    - XGBoost
    - Logistic Regression
    - Random Forest
    - AdaBoost
    - Multi-Layer Perceptron (MLP)
3. **Implementation Variants**:
    - **BareBones**: Standard scikit-learn implementation.
    - **Intelex**: scikit-learn accelerated with scikit-learn-intelex.
    - **CuPy**: Data arrays and supported models accelerated with CuPy (where possible).
4. **Benchmarking**: Each model was run on all datasets, measuring wall-clock training time.

# How the experiments were conducted
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/diagram.png" width=100% height=140%>

# Hardware used:

- **CPU**: Intel Core i5 (12th Gen)  
- **GPU**: NVIDIA GeForce GTX 1660 Super  
- **Storage**: SSD  
- **RAM**: 32 GB  

## Datasets
| Name                   | Size     | Features      | Notes                    |
|------------------------|----------|---------------|--------------------------|
| Wiretap                | 7 GB     | 115 numeric   | Largest                  |
| Loan Prediction        | 20 MB    | 12 categorical| Small                    |
| Student Performance    | 41 KB    | 10 numeric    | Smallest                 |
| Diabetes               | 636 KB   | 10 numeric    | Small                    |
| Cyberbullying Detection| 3.67 MB  | 7 text/cat.   | Medium, text-heavy       |

## Results
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/active_Wiretap_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/cyberbully%20detection_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/diabetes_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/loan%20prediction_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/student_performance_speeds.png" width=70% height=70%>

### Key Observations

#### 1. Correlation Between Dataset Size and Speed
- **Small datasets** (Student Performance, Diabetes): Most models execute in under 1 second (except MLP).
- **Medium datasets** (Loan Prediction, Cyberbullying): Performance varies widely by model and implementation. Some models (MLP) are outliers in runtime.
- **Large datasets** (Wiretap): Training can take several minutes, especially with Random Forest, AdaBoost, and MLP.

#### 2. Model Performance Trends
- **LightGBM**: Fastest across all datasets.
- **XGBoost**: Consistently fast, second only to LightGBM.
- **Logistic Regression**: Efficient on small data, but can spike in runtime for certain medium datasets.
- **Random Forest & AdaBoost**: Slow on large datasets; can be impractical for real-time use.
- **MLP**: Slowest, especially on large/medium datasets.

#### 3. Impact of Acceleration (Intelex, CuPy)
- **Intelex & CuPy**: Both can provide major speed-ups, but the effect depends on the dataset and model.
- **For small datasets**: Acceleration is noticeable, but even non-accelerated models are fast.
- **For large datasets**: Acceleration is crucial for practical runtimes, but not all models benefit equally.
- **Note**: Some models or array conversions can cause CuPy to worsen performance (e.g., Logistic Regression on Loan Prediction).

#### 4. General Takeaways
- **LightGBM is the most robust and efficient overall.**
- **CuPy and Intelex accelerations are not universally beneficial**—always profile for your use case.
- **Dataset size and model choice both impact execution time**, but model/implementation often matters more than raw size.
---

## Contributing
Feel free to fork this repository, create a new branch, and submit a pull request with your changes. Please make sure to document it well.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Diabetes Dataset (Version V1). (n.d.). [Dataset]. kaggle. https://www.kaggle.com/datasets/asinow/diabetes-dataset/data

Maree, A. (2025). Student Performance Prediction [Dataset]. In Kaggle (Version V2). Keggal. https://www.kaggle.com/datasets/amrmaree/student-performance-prediction

Mirsky, Yisroel, Doitshman, Tomer, Elovici, Yuval, & Shabtai, Asaf. (2018). Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection. In The Network and Distributed System Security Symposium (NDSS) 2018.

CyberBullying Detection Dataset. (n.d.). [Dataset]. In Kaggle (Version V2). Keggal. https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification?select=final_hateXplain.csv

Star, Ethical. (2025). Loan Prediction. Kaggle.com.. https://www.kaggle.com/datasets/ethicalstar/loan-prediction/code
