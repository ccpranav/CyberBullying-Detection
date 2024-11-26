import os
from flask import Flask, render_template, request
import re
import tensorflow as tf  # For TensorFlow/Keras model loading
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import pickle  # For loading tokenizer

# Many unnecessary warning and unwanted content displayed in the terminal is removed ny this os 
# It describes about the CPU Function
# Disable oneDNN optimization warnings and suppress TensorFlow log verbosity
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs, except errors

# Initialize Flask app
app = Flask(__name__)


# Load LSTM model and tokenizer Using Exception Handling
try:
    model = tf.keras.models.load_model('lstm_model.h5')
    # Explicitly compile the model to avoid the model compile_metrics warning (optional, if you plan to train)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

# Preprocessing function
def preprocess_comment(comment):
    comment = comment.lower()  # Lowercasing
    comment = re.sub(r'[^\w\s]', '', comment)  # Cleaning
    return comment.strip()  # Stripping

@app.route('/', methods=['GET', 'POST'])
def detect_comment():
    prediction = None  # To store prediction result
    user_input = ""  # To store user input

    if request.method == 'POST':
        if 'detect' in request.form:
            user_input = request.form['comment']  # Get user input
            processed_comment = preprocess_comment(user_input)  # Preprocess comment
            sequences = tokenizer.texts_to_sequences([processed_comment])  # Tokenize text
            padded_sequences = pad_sequences(sequences, maxlen=100)  # Padding
            prediction_label = model.predict(padded_sequences)[0][0]  # Predicting (assuming binary classification)
            prediction = "Cyberbullying" if prediction_label > 0.5 else "Non-Cyberbullying"  # Thresholding
        elif 'delete' in request.form:
            user_input = ""  # Reset input
            prediction = None  # Reset prediction

    return render_template('index.html', user_input=user_input, prediction=prediction)

if __name__ == '__main__':
    # Run the Flask app in production mode, for development use debug=False
    app.run(debug=False, use_reloader=False)  # Ensure reloader is off to avoid re-starting Flask in debug mode

