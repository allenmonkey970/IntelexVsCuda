# IntelexVsCuda

## Overview
This project aims to benchmark and compare the performance of different machine 
learning algorithms using Intel's scikit-learn-intellex and NVIDIA's Cupy-accelerated across various datasets and learning algorithms. The goal is to determine which optimization provides better performance for specific machine learning algorithms
for my college research.

- Intelex is a performance library for scikit Learn. It uses low-level optimization and multithreading to speed up performance. 
- CuPy is another performance library however, it pushes all array computations to the GPU to parrel process it. 

### Setup
**Install the required dependencies**:
   ```bash
   pip install scikit-learn
   pip install cupy-cuda
   pip install xgboost
   pip install scikit-learn-intelex
   ```
This should install on dependencies if not install pandas, warning, and numpy.

# Hardware used:

- **CPU**: Intel Core i5 (12th Gen)  
- **GPU**: NVIDIA GeForce GTX 1660 Super  
- **Storage**: SSD  
- **RAM**: 32 GB  
Here's a summary of the trends in the data:

### Dataset Overview:
1. **Wiretap**: Largest dataset (7GB) with 115 numerical columns.
2. **Loan Prediction**: Moderately sized (20MB) with 12 columns, mainly categorical.
3. **Student Performance**: Smallest dataset (41KB) with 10 columns, mainly numerical.
4. **Diabetes**: Compact dataset (636KB) with 10 numerical columns.
5. **Cyberbullying Detection**: Medium size (3.67MB) with 7 columns, text/categorical.

## Results
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/active_Wiretap_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/cyberbully%20detection_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/diabetes_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/loan%20prediction_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/student_performance_speeds.png" width=70% height=70%>


### Trends in the Data
#### 1. **Intelex**
   - Tends to have the **fastest execution times** across most datasets and algorithms.
   - Examples of notable performance:
     - **Student Performance Dataset**: Shows clear dominance with algorithms like XGBoost (112ms) and Logistic Regression (46ms).
     - **Diabetes Dataset**: Maintains the fastest speeds consistently, e.g., Logistic Regression at 61ms and XGBoost at 77ms.
     - **Cyberbullying Detection Dataset**: Performs competitively even in text-heavy datasets, e.g., Logistic Regression (1179ms).

#### 2. **CuPY**
   - Often achieves **second-best execution times**.
   - Examples of performance:
     - **Student Performance Dataset**: Decent performance in XGBoost (159ms) and Logistic Regression (152ms).
     - **Loan Prediction Dataset**: Stands out with XGBoost (3572ms) but takes longer for MLP-based approaches.
   - CuPY is particularly strong with **larger datasets** where GPU acceleration comes into play.

#### 3. **BareBones**
   - Consistently has the **slowest execution times**, especially for datasets involving larger sizes or text-heavy content.
   - Examples:
     - **Active_Wiretap Dataset**: Experiences significant lag with MLP (521,000ms) and Random Forest Classifier (218,000ms).
     - **Cyberbullying Dataset**: Slower for Logistic Regression (1157ms) and XGBoost (13,300ms).

#### General Observations
- **Intelex is ideal** for achieving consistent speed across various algorithms and datasets, particularly for numerical data.
- **CuPY performs well on larger datasets** and benefits from GPU-specific acceleration, though it lags behind Intelex in smaller datasets.
- **BareBones is less optimized** and often lags significantly, especially with computationally intense algorithms like MLP or Random Forest.

  
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
