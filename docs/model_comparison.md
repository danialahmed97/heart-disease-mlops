# Model Comparison

Stratified 80/20 split (random_state=42); 5-fold stratified GridSearchCV optimising ROC-AUC on the training portion only.

| Model | CV ROC-AUC (mean ± std) | Test Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| LogisticRegression ⭐ | 0.902 ± 0.016 | 0.885 | 0.839 | 0.929 | 0.881 | 0.966 |
| RandomForest | 0.898 ± 0.022 | 0.902 | 0.893 | 0.893 | 0.893 | 0.956 |
| XGBoost | 0.889 ± 0.023 | 0.885 | 0.862 | 0.893 | 0.877 | 0.948 |

## Best hyperparameters

- **LogisticRegression**: `{'model__C': 1, 'model__class_weight': None, 'model__penalty': 'l2'}`
- **RandomForest**: `{'model__max_depth': 3, 'model__min_samples_leaf': 3, 'model__n_estimators': 300}`
- **XGBoost**: `{'model__learning_rate': 0.03, 'model__max_depth': 2, 'model__n_estimators': 100}`

## Selection rationale

**Selected model: LogisticRegression.**

- Primary criterion: cross-validated ROC-AUC (threshold-independent ranking quality; robust on a small 303-row dataset).
- Tiebreaker: test-set recall. Models whose CV AUC is within 0.005 of the best are treated as tied and the one with the highest recall wins, since in screening a missed patient (false negative) is costlier than a false alarm.
- Selection uses CV AUC rather than test AUC so the hold-out set stays an unbiased estimate.
- With only ~61 test rows, differences of one or two patients move metrics by several points; the CV std column conveys this uncertainty.
