import numpy as np
from typing_extensions import Literal

def random_sampling(X: np.ndarray, dist: Literal['Uniform', 'Triangular', 'Geometric']= 'Uniform', size: float = 0.7):
    '''
    Parameters:
    X -> the data of N rows and m columns
    dist -> a string with the desired distribution of the sampling ['Uniform', 'Triangular', 'Geometric']
    size -> the percentage of X to be chosen

    Output:
    - A random sample containing the size*100% of the data of X, the data that was not sampled
    '''

    N,_ = X.shape
    n_samples = int(size* N)
    indexes = [i for i in range(N)]
    samples = []
    p = []
    sum_p = 0
    if dist == 'Triangular':
        a,b,c = 0, N/2, N
        for i in range(N):
            if i <= b:
                p.append((2*(i-a))/((b-a)*(c-a)))
            else:
                p.append((2*(c-i))/((c-a)*(c-b)))
            sum_p += p[i-1]
    
    if dist == 'Geometric':
        for i in range(N):
            p.append(0.5*((1-0.5)**i))
    if dist == 'Uniform':
        samples = np.random.choice(indexes, size = n_samples, replace = False)

    elif dist == 'Triangular':
        samples = np.random.choice(indexes, p = p, size = n_samples, replace = False)

    elif dist == 'Geometric':
        samples = np.random.choice(indexes, p = p, size = n_samples, replace = False)
    
    not_samples = []
    for i in indexes:
        if(i in samples):
            continue
        not_samples.append(i)
        
    return X[np.ix_(samples)], X[np.ix_(not_samples)]

