import numpy as np

def random_mix(X):
    N,_ = X.shape
    idx = np.random.permutation(N)
    X_ans = []
    for i in idx:
        X_ans.append(X[i])
    
    return np.array(X_ans)

print("hi")