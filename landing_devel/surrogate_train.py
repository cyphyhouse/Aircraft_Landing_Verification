import tensorflow as tf
import numpy as np
# print("TensorFlow Version: ", tf.__version__)
data_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/data/data.txt'
label_path = '/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/estimation_label/label.txt'
x = np.loadtxt(data_path, delimiter=',')
y = np.loadtxt(label_path, delimiter=',')
# table = [row.strip().split('\n') for row in x]

x_train = x[:8000, 1:]
y_train = y[:8000, 1:]
x_test = x[8000:, 1:]
y_test = y[8000:, 1:]

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(6,)),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(6)
])

# predictions = model(x_train[:1]).numpy()
# print(predictions)

loss_fn = tf.keras.losses.MeanSquaredError()

# print(x_train.shape)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'],
              loss_weights=[1,1,1,1000,1000,1000]
            )

model.fit(x_train, y_train, epochs=100)
print("Finished Training")
# Save the weights
model.save('/home/lucas/Research/VisionLand/Aircraft_Landing/catkin_ws/src/landing_devel/surrogate_model/model.keras')
print("Model saved.")
model.evaluate(x_test,  y_test, verbose=2)