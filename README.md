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
This should install all of the dependencies if not install Pandas, Warning, MatPlotLib, and NumPY.

# How the experiments were conducted
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/diagram.png" width=100% height=140%>

# Hardware used:

- **CPU**: Intel Core i5 (12th Gen)  
- **GPU**: NVIDIA GeForce GTX 1660 Super  
- **Storage**: SSD  
- **RAM**: 32 GB  

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

### **1. Correlation Between Dataset Size and Speed**
- **Small datasets (Student Performance, Diabetes) see execution times under ~1 second** for most models, except for MLP.  
- **Mid-sized datasets (Loan Prediction, Cyberbullying, 20MB-3.6MB)** show **big variations based on the model used**. Some implementations **spike unpredictably** (e.g., MLP in Loan Prediction).  
- **Large datasets (Wiretap, 7GB) take significantly longer execution times**:
  - **XGBoost takes ~86,000-90,000ms**
  - **Random Forest takes ~218,000-227,000ms**
  - **AdaBoost takes ~619,000-623,000ms**
  - **MLP (BareBones and Intelex) takes over 500,000ms, while CuPY slightly improves it to 284,000ms**  
  - **LightGBM remains the fastest (1,028ms)**  
---
### **2. Model Performance Trends**
- **LightGBM is the fastest** across all datasets, with execution times mostly in the range of **milliseconds to a few seconds**.  
- **XGBoost is consistently fast** but **slower than LightGBM**, particularly for larger datasets.  
- **Logistic Regression performs decently on smaller datasets** but sees **huge execution time spikes** with datasets like Loan Prediction.  
- **Random Forest is one of the slowest models**, especially for larger datasets, sometimes taking **hundreds of thousands of milliseconds**.  
- **AdaBoost is also relatively slow**, though it outperforms Random Forest in some cases.  
- **Multi-Layer Perceptron (MLP) is the slowest model in most cases**, especially on larger datasets like Loan Prediction, where it takes over **240,000ms (~4 minutes).**

---

### **3. Impact of Implementation Type (BareBones, CuPY)**
- **CuPY and Intelex often improve performance over BareBones**, but the degree of improvement varies based on the **dataset and model**.
- **For smaller datasets**, CuPY and Intelex show a significant improvement (e.g., in Student Performance and Diabetes datasets).
- **For larger datasets, the improvements are inconsistent**:
  - Example: **Loan Prediction (20MB)** – CuPY sometimes **worsens Logistic Regression performance** (37,521ms vs. 9,822ms for BareBones).  
  - Example: **Cyberbullying dataset (3.66MB)** – AdaBoost with CuPY has an abnormally high runtime (91,000ms vs. ~3,322ms with BareBones and Intelex).  
- **LightGBM is relatively stable across all implementations**, with only slight improvements from CuPY and Intelex.
- CuPY sometimes **worsens execution time**, as seen with the smaller datasets.
---

### **Key Takeaways**
1. **LightGBM is the most efficient model overall, with stable and fast execution times across all datasets.**
2. **XGBoost is a solid second choice**, especially for larger datasets, but is noticeably slower than LightGBM.  
3. **Random Forest and AdaBoost are significantly slower on large datasets**, making them less practical for high-volume data.  
4. **MLP is the worst performer** in terms of speed, struggling the most with large datasets.  
5. **CuPY and Intelex provide noticeable speed-ups,** but the effect is **not consistent across all datasets and models** because of array conversions.  
6. **Dataset size strongly correlates with execution time,** but model choice has a bigger impact than size alone.  
   - **For small datasets, even the slowest models run fast.**  
   - **For mid-sized datasets, implementation choice (CuPY/Intelex) matters more.**  
   - **For large datasets, only LightGBM remains fast, while MLP, Random Forest, and AdaBoost become impractically slow.**  
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
