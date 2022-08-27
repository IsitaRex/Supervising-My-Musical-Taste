import numpy as np

def pag(epsilon: float = 0.5, delta: float = 0.5, model: str = "Logistic Regression"):
  if(model == "Logistic Regression"):
    H = 1
  elif(model == "Decision Tree"):
    H = 1
  elif(model == "SVM linear"):
    H = 1
  elif(model == "SVM polynomial"):
    H = 1
  elif(model == "SVM rbf"):
    H = np.inf
  return (1/epsilon)*(np.log(abs(H))+np.log(1/delta))

print(pag(model = "SVM rbf"))