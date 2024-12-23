'''
Be aware that the execution time may still be significant due to the extensive number of hyperparameter combinations. 
To further reduce the execution time, you could use a smaller set of hyperparameter values or adopt a more efficient 
search strategy, such as random search or Bayesian optimization.
'''

from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
from tqdm import tqdm

# Generate fake data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter domain
a_values = [None, 'balanced']
b_values = ['gbdt', 'goss', 'dart']
c_values = list(range(10, 100, 10))  # Reduced range for faster execution
d_values = [20000]
e_values = [20]
f_values = [0.6, 0.8, 1]
g_values = [5, 10, 50, 100]
h_values = [0.01, 0.1, 0.5]

best_f1 = 0

def fit_model(a, b, c, d, e, f, g, h):
    try:
        model = LGBMClassifier(
            class_weight=a, boosting_type=b, num_leaves=c, learning_rate=h,
            subsample_for_bin=d, min_child_samples=e, colsample_bytree=f, max_depth=g,
            random_state=42
        )
        fitted = model.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return f1, (a, b, c, d, e, f, g, h)
    except Exception as e:
        print(f"Error occurred: {e}")
        return float('-inf'), (a, b, c, d, e, f, g, h)

# Perform hyperparameter search in parallel
num_cores = -1  # Use all available cores
results = Parallel(n_jobs=num_cores)(
    delayed(fit_model)(a, b, c, d, e, f, g, h)
    for a, b, c, d, e, f, g, h in tqdm(itertools.product(a_values, b_values, c_values, d_values, e_values, f_values, g_values, h_values))
)

# Find the best hyperparameters based on the highest F1 score
for f1, tuned_metrics in results:
    if f1 > best_f1:
        best_f1 = f1
        best_metrics = tuned_metrics

print('Best F1 score:', best_f1)
print('Best metrics:', best_metrics)
