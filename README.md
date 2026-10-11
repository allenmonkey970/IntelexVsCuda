# IntelexVsCuda

**Benchmarking Intel® Extension for Scikit-learn (scikit-learn-intelex) vs. CuPy-accelerated methods for machine learning workflows.**

## Overview
This project benchmarks and compares the performance of various machine learning algorithms using [Intel's scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) and [NVIDIA's CuPy](https://cupy.dev/) acceleration across multiple datasets. The goal is to determine which optimization—CPU-based (Intelex) or GPU-based (CuPy)—performs better for specific algorithms and dataset types, as part of a college research project.

- **scikit-learn-intelex**: Optimizes select scikit-learn estimators using low-level CPU instructions and multi-threading for faster performance.
- **CuPy**: GPU array library with a NumPy-compatible API; enables GPU acceleration for array computations.

## Setup

**Install the required dependencies** (use the CuPy wheel matching your CUDA version, e.g. `cupy-cuda12x`):
```bash
pip install numpy scipy pandas matplotlib scikit-learn scikit-learn-intelex xgboost lightgbm cupy-cuda12x
```

**Put the dataset CSVs in `data/`** (git-ignored; see [Acknowledgments](#acknowledgments) for sources):
`Active_Wiretap_dataset.csv`, `Active_Wiretap_labels.csv`, `final_hateXplain.csv`, `diabetes_dataset.csv`, `Loan Prediction.csv`, `student_performance_dataset.csv`.

**Run the benchmark, then build the tables and charts:**
```bash
python benchmark.py --dataset all --impl all --repeats 9 --warmup 1   # appends to results/runs.csv
python results.py                                                     # writes results/summary.md, summary.csv and charts
```
Use `--dataset diabetes`, `--impl Intelex` or `--models "Random Forest Classifier"` to run a subset.

The notebooks in `models/` are the original exploratory code. Published numbers come from `benchmark.py`.

## Experiment Methodology

1. **Preprocessing** (`bench/data.py`): each dataset has a single preprocessing pipeline, shared by all implementations. Feature selection, TF-IDF and scaling are fitted on the training split only. Splits are stratified with a fixed seed.
2. **Models** (`bench/models.py`): the benchmark runs only the classifiers that scikit-learn-intelex actually patches, verified against `sklearnex.get_patch_map()`, so every Intelex measurement reflects a real oneDAL path. One hyperparameter set per model, the same on every dataset and implementation:
    - Random Forest
    - Logistic Regression
    - Extra Trees
   LightGBM and XGBoost (not scikit-learn) and AdaBoost and the MLP (scikit-learn, but not patched by sklearnex) were removed for that reason.
3. **Implementation Variants**:
    - **Intelex**: `patch_sklearn()` is applied in a fresh process *before* any estimator is imported, so each model runs on the oneDAL backend. The `Backend` column records which library actually ran each model (`sklearnex.*` when patched), so a fallback is visible rather than silently counted as "Intelex".
    - **CuPy**: data is moved to the GPU with CuPy. None of these three estimators has a CuPy/GPU training path, so the data is copied back to the host and they train on the CPU with stock scikit-learn. That round trip is measured as transfer time, so this variant is effectively *stock scikit-learn + transfer overhead*.
    - **BareBones** (stock scikit-learn, no transfer) is still available via `--impl BareBones` if an untaxed CPU baseline is wanted; the published run below compares Intelex against CuPy.
4. **Measurement protocol** (`bench/harness.py`):
    - **Warm-up**: 1 discarded run per model, so first-call overhead is excluded. This covers library loading, JIT/kernel compilation and the CUDA context, which is also created before any timer starts.
    - **Repeats**: 9 measured runs per model (configurable). Tables report **mean ± standard deviation**.
    - **Separate timers**: **transfer** (host↔GPU copies), **fit** (training) and **predict** (inference on the test split). GPU work is synchronised before each timer stops.
    - **Quality**: **accuracy** and **macro F1** on the held-out test split, reported next to every time. A speed-up that comes with a drop in F1 is not counted as a win.
    - **Speedup**: baseline mean time ÷ implementation mean time (> 1 = faster), reported for fit time and for end-to-end time (transfer + fit + predict).
    - Each implementation runs in its own Python process, so the Intelex patch cannot leak into the other runs.

# How the experiments were conducted
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/diagram.png" width=100% height=140%>

# Hardware used:

- **CPU**: Intel Core i9-10900F (10 cores / 20 threads)  
- **GPU**: NVIDIA GeForce RTX 4060 (8 GB)  
- **Storage**: SSD  
- **RAM**: 32 GB  

## Datasets
| Name                   | Size     | Features      | Notes                    |
|------------------------|----------|---------------|--------------------------|
| Wiretap                | 7 GB     | 115 numeric   | Largest                  |
| Loan Prediction        | 20 MB    | 12 categorical| Small                    |
| Student Performance    | 41 KB    | 10 numeric    | Smallest                 |
| Diabetes               | 3.81 MB  | 8 mixed       | Medium                   |
| Cyberbullying Detection| 3.67 MB  | 7 text/cat.   | Medium, text-heavy       |

## Results

Full per-cell tables (mean ± sd for transfer, fit, predict and total time, plus accuracy and macro F1 and the backend that actually ran) are in [`results/summary.md`](results/summary.md), produced by `python results.py`. The run below is all five datasets × the three patched models × 9 repeats, with Active Wiretap at full scale (2.28M rows).

### Intelex speedup vs CuPy (end-to-end: transfer + fit + predict)

Each cell is **CuPy total ÷ Intelex total**, so > 1 means Intelex is faster. Because none of these models has a GPU training path, "CuPy" here is stock scikit-learn plus the host↔GPU round trip, so this reads as *oneDAL vs stock scikit-learn*.

| Model | Wiretap (2.28M) | Cyberbullying (sparse) | Diabetes | Loan | Student |
|---|---|---|---|---|---|
| Random Forest | **3.8×** | 1.0× | **2.6×** | 0.94× | **36×** |
| Logistic Regression | **6.6×** | 1.0× | **3.3×** | 0.62× | 0.71× |
| Extra Trees | 1.2× | 1.0× | 0.60× | 1.2× | **26×** |

**What the numbers show:**

1. **oneDAL wins clearly on dense numeric data**, most of all for Logistic Regression and Random Forest: 6.6× and 3.8× on the full 2.28M-row Wiretap capture, and 3.3× / 2.6× on Diabetes.
2. **On sparse TF-IDF input (Cyberbullying) Intelex gives no speedup** — sklearnex falls back to stock scikit-learn for sparse data, so all three models tie.
3. **On tiny data (Student, ~700-row test split) the tree models are 26–36× faster** under Intelex, because oneDAL's fixed per-call overhead is far lower than stock scikit-learn's.
4. **Extra Trees is inconsistent** — oneDAL is slower than stock on Diabetes (0.60×) and a wash elsewhere, so it is not a blanket win.
5. **Accuracy and macro F1 match across implementations to ~4 decimals** on every cell, so the Intelex speedups come with no quality regression.

### Charts

End-to-end time per model (log scale, mean ± sd error bars), regenerated by `python results.py`:

<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/active_Wiretap_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/cyberbully%20detection_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/diabetes_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/loan%20prediction_speeds.png" width=70% height=70%>
<img src="https://github.com/allenmonkey970/IntelexVsCuda/blob/master/results/student_performance_speeds.png" width=70% height=70%>

<details>
<summary>Legacy experiment (superseded)</summary>

> The numbers below come from the **original experiment** ([`PerformanceSpeeds.csv`](PerformanceSpeeds.csv), full table in [`results/legacy_summary.md`](results/legacy_summary.md)). They are single runs of training time only, with no warm-up. Read them with the caveats listed after the table.

### Legacy speedup factors vs BareBones (training time, single run)

Each cell is **Intelex × / CuPy ×**. Speedup = BareBones time ÷ variant time, so > 1 is faster and < 1 is slower.

| Model | Wiretap (7 GB) | Loan (20 MB) | Cyberbullying (3.7 MB) | Diabetes (636 KB) | Student (41 KB) |
|---|---|---|---|---|---|
| LightGBM | 1.01 / 1.01 | 1.07 / 1.10 | 1.09 / 0.89 | 2.03 / 0.75 | 3.00 / 0.73 |
| XGBoost | 1.02 / 0.98 | 1.07 / 1.19 | 1.00 / **9.70†** | 3.52 / 2.01 | 2.41 / 1.70 |
| Logistic Regression | 0.97 / 1.02 | 0.99 / 0.26 | 0.98 / 0.85 | 1.02 / 0.53 | 1.09 / 0.33 |
| Random Forest | 1.00 / 0.96 | 0.95 / 0.49 | 0.99 / 0.99 | 1.01 / 0.93 | 0.98 / 0.59 |
| AdaBoost | 1.00 / 1.00 | 0.95 / 0.85 | 1.00 / **0.04‡** | 1.03 / 0.90 | 1.16 / 0.21 |
| MLP | 1.00 / 1.83 | 1.00 / 0.89 | 0.97 / 1.27 | 0.97 / 0.87 | 1.01 / 0.45 |

† Faster, but the model is wrong: see the accuracy table below. ‡ 91 s vs 3.3 s: that cell converted the sparse TF-IDF matrix to a dense array before moving it to the GPU.

### Accuracy in the legacy runs (from the notebook outputs)

| Case | Speed | Quality |
|---|---|---|
| Cyberbullying, XGBoost, CuPy vs BareBones | 9.70× faster (1.4 s vs 13.3 s) | accuracy **0.36 vs 0.64**, macro F1 **0.22 vs 0.62**. The CuPy cell used a binary objective (`binary:logistic`) on a 3-class target. |
| Loan, AdaBoost, BareBones | 7.0 s | accuracy 0.876, but positive-class F1 **0.00** (macro F1 0.47). It predicts "no risk" for everyone, and 0.876 is exactly the majority-class share (44,147 / 50,400). |
| Student performance, XGBoost / LR / RF / AdaBoost, all variants | 25–275 ms | accuracy **1.00** everywhere (MLP: 0.63). Perfect scores on a 142-row test set suggest a feature that leaks the target. Check this before drawing conclusions from this dataset. |

### What the legacy numbers do and do not show

1. **Intelex had no measurable effect on the scikit-learn models: 0.95×–1.16× (median 1.00×).** In the original notebooks, `patch_sklearn()` ran *after* the estimators had already been imported, so the "Intelex" cells most likely ran stock scikit-learn. The new harness fixes this and records the backend that actually ran.
2. **The largest "Intelex" speedups (2.0×–3.5×, on XGBoost and LightGBM for Diabetes and Student) are run-to-run noise.** sklearnex does not touch XGBoost or LightGBM, so the same code ran in both cells. A 2–3.5× gap between identical code is what single, un-warmed runs produce, e.g. from library loading or one-off setup on the first call. This is why the new protocol uses a warm-up plus repeated runs.
3. **CuPy was faster in 7 of 30 cells, slower in 17, and within ±5% in 6 (median 0.89×).** For every model except XGBoost, "CuPy" only meant copying data to the GPU and back, so it could only add time. Its one real GPU result (XGBoost, Cyberbullying, 9.70×) came with a model that is worse than always predicting the majority class (0.36 vs 0.37 accuracy).
4. **XGBoost used the GPU (`device="cuda"`) in all three legacy variants**, so its legacy numbers do not compare CPU with GPU. The harness now runs XGBoost on the CPU for BareBones and Intelex.
5. **LightGBM was the fastest model on only 2 of 5 datasets** (Wiretap: 1.0 s vs 88 s for the next fastest model, XGBoost; Loan: 2.5 s vs 4.2 s). Logistic Regression was fastest on Cyberbullying (1.2 s) and Diabetes (62 ms), and AdaBoost on Student (29 ms).
6. **Dataset size dominates absolute time.** On Wiretap, training took 88 s (XGBoost) to 622 s (AdaBoost), with LightGBM at 1.0 s. On Diabetes and Student, every model trained in under 1.4 s.
7. The Student Performance notebook has no LightGBM cell, so that CSV row cannot be reproduced from the repository.

</details>

---

## Contributing
Feel free to fork this repository, create a new branch, and submit a pull request with your changes. Please make sure to document it well.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Mustafa, Mohammed. (n.d.). Diabetes Prediction Dataset [Dataset]. In Kaggle. https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Maree, A. (2025). Student Performance Prediction [Dataset]. In Kaggle (Version V2). Keggal. https://www.kaggle.com/datasets/amrmaree/student-performance-prediction

Mirsky, Yisroel, Doitshman, Tomer, Elovici, Yuval, & Shabtai, Asaf. (2018). Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection. In The Network and Distributed System Security Symposium (NDSS) 2018.

CyberBullying Detection Dataset. (n.d.). [Dataset]. In Kaggle (Version V2). Keggal. https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification?select=final_hateXplain.csv

Star, Ethical. (2025). Loan Prediction. Kaggle.com.. https://www.kaggle.com/datasets/ethicalstar/loan-prediction/code
