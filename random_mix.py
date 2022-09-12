import numpy as np

def random_mix(X):
    '''
    Input:
    - X: a numpy array of size (N,m)

    Output:
    - A new permutation of the numpy array over dimension N
    '''
    N,_ = X.shape
    idx = np.random.permutation(N)
    X_ans = []
    for i in idx:
        X_ans.append(X[i])
    
    return np.array(X_ans)