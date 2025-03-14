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
   pip install tensorflow
   ```
This should install on dependencies if not install pandas, warning, and numpy.

# Hardware used:

- **CPU**: Intel Core i5 (12th Gen)  
- **GPU**: NVIDIA GeForce GTX 1660 Super  
- **Storage**: SSD  
- **RAM**: 32 GB  

## Results

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
