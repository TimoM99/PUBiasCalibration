Learning from biased positive-unlabeled data via threshold calibration
======================================================================

This repository contains code to run the methods discussed in 'Learning from biased positive-unlabeled data via threshold calibration'

Installation
------------
From source:
```bash
git clone https://github.com/TimoM99/PUBiasCalibration.git
cd PUBiasCalibration
pip install .
```

Usage
-----
The source folder gives access to the different methods used in this paper.

- `basic.py` : Model using fully labeled data.
- `LBE.py` : The LBE model by Gong et al. [1]
- `PGlin.py` : The PGLin model by Gerych et al. [2]
- `PUe.py` : The PUe model by Wang et al. [3]
- `PUSB.py` : The PUSB model by Kato et al. [4]
- `SAREM.py` : The SAR-EM model by Bekker et al. [5]
- `threshold.py` : Our proposed NTC + threshold model, NTC-tMI. [6]

One could also use the threshold optimization algorithm in isolation using `threshold_optimizer.py`.

Examples
--------
```python
from sklearn.linear_model import LogisticRegression
import numpy as np
from PUBiasCalibration.threshold_optimizer import ThresholdOptimizer

#X, s and X_test are numpy arrays of data.
ntc = LogisticRegression()
ntc.fit(X, s) 
lin_pred = ntc.decision_function(X)

t = ThresholdOptimizer(3, 100).find_threshold(lin_pred[s=0]) #k=3, m=100

test_lin_pred = ntc.decision_function(X_test)
test_pred = np.where(test_lin_pred > t, 1, 0)
```
```python
from PUBiasCalibration.Models.threshold import PUthreshold

#X, s and X_test are numpy arrays of data.
model = PUthreshold()
model.fit(X, s)

model.predict_proba(X_test)
```

[1] Gong, C., Wang, Q., Liu, T., Han, B., You, J., Yang, J., & Tao, D. (2021). Instance-dependent positive and unlabeled learning with labeling bias estimation. IEEE transactions on pattern analysis and machine intelligence, 44(8), 4163-4177.
[2] Gerych, W., Hartvigsen, T., Buquicchio, L., Agu, E., & Rundensteiner, E. (2022, June). Recovering the propensity score from biased positive unlabeled data. In Proceedings of the AAAI conference on artificial intelligence (Vol. 36, No. 6, pp. 6694-6702).
[3] Wang, X., Chen, H., Guo, T., & Wang, Y. (2023). Pue: Biased positive-unlabeled learning enhancement by causal inference. Advances in Neural Information Processing Systems, 36, 19783-19798.
[4] Kato, M., Teshima, T., & Honda, J. (2019, May). Learning from positive and unlabeled data with a selection bias. In International conference on learning representations.
[5] Bekker, J., Robberechts, P., & Davis, J. (2019, September). Beyond the selected completely at random assumption for learning from positive and unlabeled data. In Joint European conference on machine learning and knowledge discovery in databases (pp. 71-85). Cham: Springer International Publishing.
[6] Teisseyre, P., Martens, T., Bekker, J., & Davis, J. (2025) Learning from biased positive-unlabeled data via threshold calibration. In Proceedings of AISTATS 2025.

