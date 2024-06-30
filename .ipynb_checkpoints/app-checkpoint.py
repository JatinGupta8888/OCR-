from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import cv2 
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle

stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

with open('count_vectorize.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('xgb_cv.pkl', 'rb') as f:
    xgb_cv = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'jpg', 'jpeg', 'png'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to extract text from an image using Tesseract OCR
def extract_text(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

# Function to preprocess text for classification
def preprocess_text(text):
    cleaned_text = clean_text(text)
    stemmed_tokens = [stemmer.stem(word) for word in word_tokenize(cleaned_text) if word.lower() not in stop_words]
    return ' '.join(stemmed_tokens)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        # If the user does not select a file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract text from the uploaded image
            extracted_text = extract_text(file_path)

            # Preprocess the extracted text for classification
            preprocessed_text = preprocess_text(extracted_text)

            # Use CountVectorizer to transform the text
            vectorized_text = cv.transform([preprocessed_text])

            # Predict class using the XGBoost model
            predicted_class_index = xgb_cv.predict(vectorized_text)[0]

            predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

            return render_template('result.html', filename=filename, extracted_text=extracted_text, predicted_class=predicted_class)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
