# ML Assignment 2
**Programme:** M.Tech (AIML)  
**Subject:** Machine Learning  
**Assignment:** Multiple Classification Models  
**BITS ID:** 2026CT05128  

## 1. Problem statement

The objective is to build and compare five supervised machine-learning classification models to predict whether an individual's annual income is above $50,000 (`>50K`) or at/below $50,000 (`<=50K`) using demographic and employment-related attributes. The models are Logistic Regression, Decision Tree, K-Nearest Neighbors (kNN), Gaussian Naive Bayes, and Random Forest.

## 2. Dataset description

- **Dataset:** Adult dataset
- **Source:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/2/adult
- **UCI dataset ID:** 2
- **Target:** `income`
- **Task:** Binary classification
- **Predictors:** 14 attributes: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.
- **Feature types:** Six numerical and eight categorical attributes in the standard dataset.
- **Data quality:** The dataset contains missing values represented in the source data by `?`. The script converts these markers to missing values, reports their counts, and imputes them inside the training pipeline.
- **Class balance:** The standard dataset is imbalanced, with the `<=50K` class substantially larger than `>50K`. The script prints exact counts and percentages for the retrieved version.

The dataset was selected because it exceeds the assignment minimum of 1,000 instances and 12 predictors, includes both numerical and categorical data, contains missing values, and presents a realistic imbalanced binary-classification problem.

## 3. Data Preprocessing

1. Convert source `?` and blank markers to missing values.
2. Identify and remove duplicate complete records.
3. Impute numerical missing values with the training-set median.
4. Impute categorical missing values with the training-set most frequent category.
5. One-hot encode categorical features, ignoring unseen test categories.
6. Standardize numerical features.
7. Split 80% training and 20% testing using `stratify=y` and `random_state=42`.

## 4. GitHub Repository Link

**Repository URL:** https://github.com/2026CT05128/2026CT05128

## 5. Models used and evaluation

All five models use the same stratified 80:20 train-test split and the same preprocessing design. The random seed is 42. Imputers, encoders, and scalers are fitted only on the training portion through a scikit-learn `Pipeline` and `ColumnTransformer`.

### Model configurations

| Model | Configuration |
|---|---|
| Logistic Regression | max_iter=2000, class_weight=balanced |
| Decision Tree | max_depth=12, min_samples_split=10, class_weight=balanced |
| kNN | n_neighbors=5, weights=distance |
| Gaussian Naive Bayes | GaussianNB |
| Random Forest | n_estimators=300, min_samples_split=5, class_weight=balanced_subsample |

### Evaluation metrics

The positive class is `>50K`.

- **Accuracy:** proportion of all predictions that are correct.
- **AUC:** area under the ROC curve, calculated using predicted probability for `>50K`.
- **Precision:** proportion of predicted `>50K` cases that are truly `>50K`.
- **Recall:** proportion of actual `>50K` cases correctly identified.
- **F1:** harmonic mean of precision and recall.
- **MCC:** Matthews correlation coefficient, which considers all four confusion-matrix outcomes and is useful for imbalanced classes.

### Comparison table

<img width="757" height="175" alt="image" src="https://github.com/user-attachments/assets/b80e7d0d-d61a-4262-9d5a-39fb16b01270" />


### Graph representation

The program generates:
- `outputs/all_metrics_comparison.png` – grouped bar chart for all six metrics and five models.
- `outputs/metrics_heatmap.png` – heatmap of metric scores.
- One confusion-matrix image per model in `outputs/`.

Include these generated images in the PDF report.

### Observations on model performance

<img width="1158" height="320" alt="image" src="https://github.com/user-attachments/assets/ece3dc30-c9d0-4d98-92c7-3a581ee20aa7" />

**Winner selection:** Based on the experimental evaluation of all five classification algorithms, Random Forest emerged as the best-performing model for the Adult Census Income dataset. It achieved the highest accuracy of 85.69%, AUC of 91.25%, F1 score of 0.7032, and MCC score of 0.6090.

The model also achieved the highest precision of 69.85%, demonstrating better reliability in predicting individuals earning above $50,000. Although Gaussian Naive Bayes achieved the highest recall of 93.36%, its lower precision and accuracy indicate a larger number of false-positive predictions.

Random Forest provides a better balance between precision and recall and demonstrates stronger overall classification performance. Therefore, it is selected as the overall winner for this dataset.

## Execution instructions

### 1. Create and activate an environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / BITS Virtual Lab:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the experiment

From the project root:
```bash
python model/train_models.py
```

The script retrieves the dataset from UCI. An internet connection is required for the first download.

### 4. Generated files

- `test_data.csv` – held-out 20% test partition, including the target for evaluation.
- `saved_models/` – five fitted preprocessing-and-classification pipelines saved as Joblib files.
- `outputs/evaluation_metrics.csv` – exact evaluation metrics.
- `outputs/all_metrics_comparison.png` – metric comparison graph.
- `outputs/metrics_heatmap.png` – metric heatmap.
- `outputs/*_confusion_matrix.png` – confusion matrices.
- `outputs/*_classification_report.txt` – per-model classification reports.

## Reproducibility and data leakage prevention

The split uses `random_state=42` and stratification. Each model is trained through a pipeline. Therefore, imputation, one-hot encoding and scaling are learned from training data only; the test set is used only for final evaluation. The same held-out test partition is used for all five models.

## Limitations and ethical considerations

The Adult dataset reflects historical US census data. Its demographic attributes and income labels may encode social and historical biases. Predictions should not be used to make consequential decisions about individuals. The model is an educational experiment, not a fair or validated income-assessment tool.

## Repository structure

```text
adult-income-classification/
├── README.md
├── requirements.txt
├── test_data.csv                 # generated after running the script
├── model/
│   └── train_models.py
├── saved_models/                 # generated after running the script
└── outputs/                      # generated after running the script
    ├── evaluation_metrics.csv
    ├── all_metrics_comparison.png
    ├── metrics_heatmap.png
    ├── *_confusion_matrix.png
    └── *_classification_report.txt
```
