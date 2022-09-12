class MLP(object):
  # Multilayer Perceptron class

  def __init__(self, 
    layers: List[Layer],
    learning_rate: float = 0.1):

    self.layers = layers # List of layers
    self.learning_rate = learning_rate # Learning rate
    self.errors  = [] # List to store errors
  
  def forward(self, stimulus: np.ndarray)->np.ndarray:
    '''
    Forward propagation
    '''
    for layer in self.layers:
      stimulus = layer.forward(stimulus)
    return stimulus


  def backward(self,  y: np.ndarray)->None:
    '''
    Backward propagation
    '''
    local_gradient = self.layers[-1].output - y
    for layer in reversed(self.layers):
      local_gradient = layer.backward(local_gradient)

  def update_weights(self)-> None:
    '''
    Update weights
    '''
    for layer in self.layers:
      layer.update(self.learning_rate)

  def train(self, x: np.ndarray, y: np.ndarray, epochs: int)->None:
    '''
    Train MLP
    '''
    self.all_gradients = []
    for epoch in range(epochs):
      y_output = self.forward(x)
      error = abs(y - y_output)
      self.backward(y)
      self.all_gradients.append(self.get_gradients())
      self.update_weights()

      if epoch%100 == 0:
        print("Epoch: ", epoch, "Error: ", np.mean(error))
      self.errors.append(np.mean(error))

  def get_gradients(self)->List[np.ndarray]:
    '''
    Get gradients
    '''
    gradients = []
    for layer in self.layers:
      gradients.append(np.mean(layer.weights_grad))
    return gradients
    
  def predict(self, x)->np.ndarray:
    y_output = self.forward(x)
    return y_output