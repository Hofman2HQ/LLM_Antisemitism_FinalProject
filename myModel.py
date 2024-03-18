import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, TextVectorization
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('labeled_data.csv')

# Encode the class labels
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the TextVectorization layer
max_features = 10000  # Maximum vocabulary size
sequence_length = 100  # Maximum sequence length
vectorize_layer = TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)

# Adapt the TextVectorization layer to the training data
train_texts = train_df['tweet'].values
vectorize_layer.adapt(train_texts)

# Vectorize the training and testing data
X_train = vectorize_layer(train_texts)
y_train = train_df['class'].values
X_test = vectorize_layer(test_df['tweet'].values)
y_test = test_df['class'].values

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features + 1, 64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=11, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predict on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()