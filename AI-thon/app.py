from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None
label_encoder = None

# Get the current working directory

current_directory = os.getcwd()

# Define the name of your CSV file
file_name = r'music.csv'

# Combine the current directory and file name to create the full file path
file_path = os.path.join(current_directory, file_name)

# Check if the file exists
if os.path.exists(file_path):
    # Load the dataset from the CSV file
    df = pd.read_csv(file_path)

    # Prepare the features (moods) and target variable (music preferences)
    X = df['Mood']
    y = df['Preference']

    # Encode the target variable using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Create a logistic regression model
    model = LogisticRegression()

    # Train the model on the entire dataset
    model.fit(X_vectorized, y_encoded)

@app.route('/', methods=['GET', 'POST'])
def index():
    music_recommendation = None

    if request.method == 'POST':
        user_input = request.form['mood']

        # Check if the vectorizer is fitted before transforming
        #if vectorizer is not None and hasattr(vectorizer, 'transform'):
            # Vectorize the user's input using the same vectorizer
        user_input_vectorized = vectorizer.transform([user_input])

            # Predict the music recommendation
        prediction = model.predict(user_input_vectorized)

            # Decode the predicted label using the label encoder
        music_recommendation = label_encoder.inverse_transform(prediction)[0]

    return render_template('index.html', music_recommendation=music_recommendation)

if __name__ == '__main__':
    app.run(debug=True)
