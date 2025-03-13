# IntelexVsCuda

## Overview
This project aims to benchmark and compare the performance of different machine 
learning algorithms using Intel's scikit-learn-intellex and NVIDIA's Cupy-accelerated across various datasets and learning algorithms. The goal is to determine which optimization provides better performance for specific machine learning algorithms
for my college research.
### Setup
**Install the required dependencies**:
   ```bash
   conda install scikit-learn
   conda install -c conda-forge cupy
   conda install -c conda-forge py-xgboost-gpu
   conda install scikit-learn-intelex 
   ```
This should install on dependencies if not install pandas, warning, and numpy.

# Hardware used:

- **CPU**: Intel Core i5 (12th Gen)  
- **GPU**: NVIDIA GeForce GTX 1660 Super  
- **Storage**: SSD  
- **RAM**: 32 GB  

## Results

### Wiretap Dataset (115 numerical columns)
- **XGBoost:** Intelex performed best (87,000 ms; CuPY data unavailable).
- **Logistic Regression:** CuPY was the fastest (117,000 ms).
- **Random Forest Classifier:** CuPY outperformed others (121,000 ms).

### Student Performance Dataset (10 mainly numerical columns)
- **XGBoost:** CuPY was the fastest (502 ms).
- **Logistic Regression:** Intelex was the fastest (410 ms).
- **Random Forest Classifier:** Intelex excelled (335 ms).

### Diabetes Dataset (17 numerical columns)
- **XGBoost:** CuPY led (527 ms).
- **Logistic Regression:** Intelex performed best (444 ms).
- **Random Forest Classifier:** Intelex was the fastest (926 ms).

### Loan Prediction Dataset (13 mainly categorical columns)
- **XGBoost:** CuPY outperformed others (3,699 ms).
- **Logistic Regression:** Intelex was the fastest (9,684 ms; CuPY was significantly slower).
- **Random Forest Classifier:** Intelex was the fastest (154,000 ms; CuPY data unavailable).

### Cyberbully Detection Dataset (7 text and categorical columns)
- **XGBoost:** CuPY was the fastest (1,752 ms).
- **Logistic Regression:** Intelex performed best (1,493 ms).
- **Random Forest Classifier:** Intelex slightly outpaced others (23,764 ms).

---

### Summary of Algorithm Trends
- **XGBoost:** CuPY is consistently the best for most datasets.
- **Logistic Regression:** Intelex consistently leads, offering the best speed across datasets.
- **Random Forest Classifier:** Intelex is the fastest for most datasets.

## Contributing
Feel free to fork this repository, create a new branch, and submit a pull request with your enhancements. Please make sure to add test cases for any new features or changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Diabetes Dataset (Version V1). (n.d.). [Dataset]. kaggle. https://www.kaggle.com/datasets/asinow/diabetes-dataset/data

Maree, A. (2025). Student Performance Prediction [Dataset]. In Kaggle (Version V2). Keggal. https://www.kaggle.com/datasets/amrmaree/student-performance-prediction

Mirsky, Yisroel, Doitshman, Tomer, Elovici, Yuval, & Shabtai, Asaf. (2018). Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection. In The Network and Distributed System Security Symposium (NDSS) 2018.

CyberBullying Detection Dataset. (n.d.). [Dataset]. In Kaggle (Version V2). Keggal. https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification?select=final_hateXplain.csv

Star, Ethical. (2025). Loan Prediction. Kaggle.com.. https://www.kaggle.com/datasets/ethicalstar/loan-prediction/code
