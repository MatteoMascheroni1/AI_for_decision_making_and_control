import numpy as np
from hmmlearn import hmm

np.random.seed(42)
componets = 6
delta = 0.05
startprob = np.array([1, 0, 0, 0, 0, 0])
transtprob = np.array([[0.1, 0.9, 0, 0, 0, 0],
                        [0.2, 0, 0.1+delta, 0, 0, 0.7-delta],
                        [0, 0.5, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0.5, 0, 0],
                        [0, 0, 0.1, 0, 0.4, 0.5],
                        [0, 0.2, 0, 0, 0.7, 0.1]])
emissionprob = np.array([[0.25, 0.25, 0.25, 0.25],
                         [0.4, 0.3, 0.2, 0.1],
                         [0.4, 0.4, 0.1, 0.1],
                         [0.3, 0.3, 0.3, 0.1],
                         [0.2, 0.2, 0.2, 0.4],
                         [0.2, 0.2, 0.3, 0.3]
                         ])

model = hmm.CategoricalHMM(n_components=componets)
model.startprob_=startprob
model.transmat_ = transtprob
model.emissionprob_ = emissionprob


O,Q = model.sample(100)
print(f"Observation matrix: {O}")
print(f"State sequence: {Q}")
