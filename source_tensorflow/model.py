import tensorflow as tf

class BinaryClassifier(tf.keras.Model):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in Tensorflow, use BinaryCrossentropy loss.
    documentation: https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
    """
    def __init__(self, input_features, hidden_dim, output_dim, dropout_factor):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """        
        super(BinaryClassifier, self).__init__()
        self.dense1 = tf.keras.layers.Dense(input_features, activation="relu")
        self.dense2 = tf.keras.layers.Dense(hidden_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_factor)
        self.dense3 = tf.keras.layers.Dense(output_dim, activation="sigmoid")
        
    
    def call(self, inputs, training=False):
        """
        Perform a forward pass of our model on input features, x.
        :param training: weather the training is active or not.
        :param inputs: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        x = self.dense1(inputs)
        x = self.dense2(x)
        if training:
            x = self.dropout(x, training=training)
        x = 
        return self.dense3(x) 