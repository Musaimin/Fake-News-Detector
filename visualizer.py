import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load your saved model
model = load_model('./Models/simple_rnn_model.keras')

# Create a graph plot of your model
plot_model(model, to_file='model_plot.png', show_shapes=True)

# Load your saved model
model = load_model('./Models/lstm_model.keras')

# Create a graph plot of your model
plot_model(model, to_file='lstm_model_plot.png', show_shapes=True)
