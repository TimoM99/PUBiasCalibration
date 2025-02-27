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
- `threshold.py` : Our proposed NTC + threshold model, NTC-tMI.

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
