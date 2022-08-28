from math import log2
import numpy as np

def pag(epsilon: float = 0.5, delta: float = 0.5, model: str = "Logistic Regression", num_features: int = 3, depth_tree: int = 3):
  if(model == "Logistic Regression"):
    H = num_features + 1
  elif(model == "Decision Tree"):
    return (np.log(2)/(2*epsilon*epsilon))*((2**depth_tree - 1)*(1+log2(num_features))+1+np.log(1/delta))
  elif(model == "SVM linear"):
    H = num_features
  elif(model == "SVM polynomial"):
    H = 6
  elif(model == "SVM rbf"):
    H = np.inf
  return (1/epsilon)*(np.log(abs(H))+np.log(1/delta))

print(pag(model = "Logistic Regression"))