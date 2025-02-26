#This file is used to generate the plot in appendix of the paper.
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from src.threshold_optimizer import ThresholdOptimizer
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

nb_labeled = 2000
nb_unlabeled_positive = 1000
nb_unlabeled_negative = 1000

labeled = np.random.normal(-5, 0.3, nb_labeled)
unlabeled_positive = np.random.normal(3, 0.5, nb_unlabeled_positive)
unlabeled_negative = np.random.normal(5, 0.5, nb_unlabeled_negative)


X = np.concatenate([labeled, unlabeled_positive, unlabeled_negative])
y = np.concatenate([np.ones(nb_labeled), np.zeros(nb_unlabeled_positive), np.zeros(nb_unlabeled_negative)])

model = LogisticRegression()
model.fit(X.reshape(-1, 1), y)
scores = model.decision_function(X.reshape(-1, 1))
scores_unlabeled = scores[nb_labeled:]

optimizer = ThresholdOptimizer(k=3, n=100)

t = optimizer.find_threshold(scores)
t_unlabeled = optimizer.find_threshold(scores_unlabeled)


sns.kdeplot(scores[:nb_labeled], color='magenta', label='Labeled')
sns.kdeplot(scores[nb_labeled: nb_labeled + nb_unlabeled_positive:], color='blue', label='Unlabeled Positive')
sns.kdeplot(scores[nb_labeled + nb_unlabeled_positive:], color='orange', label='Unlabeled Negative')
plt.axvline(x=t, color='green', linestyle='-', label='Threshold over all Z')
plt.axvline(x=t_unlabeled, color='red', linestyle='-', label='Threshold over unlabeled Z')
plt.xlabel('Z')
plt.legend()
plt.savefig('labeled_vs_unlabeled_MI.pdf')


