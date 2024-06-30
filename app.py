from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import cv2
import re
from textblob import TextBlob
import pyttsx3
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

with open('RF.pkl', 'rb') as f:
    RF = pickle.load(f)

with open('CV.pkl', 'rb') as f:
    CV = pickle.load(f)
    
def rf_text_cleaning(text):
    text = text.lower().strip()
    
    pattern = re.compile('\W')
    text = re.sub(pattern, ' ', text).strip()

    pattern = re.compile(r'\d+')
    text = re.sub(pattern, '', text)
    
    text = re.sub(r'\s+', ' ', text).strip() # Removing extra whitespace
    
    return text

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

def stopword_stemming(text):
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def remove_top_lines_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    dilate = cv2.dilate(binary, kernel, iterations=1)
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    for c in contour:
        x, y, w, h = cv2.boundingRect(c)
        if y < 100 and h < img.shape[0] // 4 and w > img.shape[1] // 2:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return img

def desired_text(img):
    results = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    dilate = cv2.dilate(binary, kernel, iterations=1)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > img.shape[0] / 7 and w > 30 and y < img.shape[0] / 1.14:
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            roi = img[y:y + h, x:x + h]
            ocr_result = pytesseract.image_to_string(roi)
            ocr_result = ocr_result.split("\n")
            for item in ocr_result:
                results.append(item)
    return img, ' '.join(results)

def preprocess_correct_text(text):
    pattern = r'[\[\]{}()@#&$%*^!<>?;:/\\|+-]'
    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    blob = TextBlob(cleaned_text)
    corrected_text = str(blob.correct())
    return corrected_text
###################################################################3

@app.route('/')
def homepage():
    return render_template('upload.html')


@app.route('/results', methods=['GET', 'POST'])
def upload_file():
    file = request.files.get('file')
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = cv2.imread(file_path)
        img = remove_top_lines_from_image(img)
        img, text = desired_text(img)
        processed_image_path = os.path.join(app.config['STATIC_FOLDER'],filename)
        cv2.imwrite(processed_image_path, img)
        extracted_text = preprocess_correct_text(text)

        classes = {
            1:'World News',
            2:'Sports News',
            3:'Business',
            4:'Sci/Tech News',
        }
        article = rf_text_cleaning(extracted_text)
        article = stopword_stemming(article)
        vector = CV.transform([article])
        prediction = RF.predict(vector)

        text_class = classes[prediction[0]]

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 130)
        engine.setProperty('volume', 1)
        string = filename[:-4]
        string = string+'.mp3'
        audio_file = string
        processed_image_path = os.path.join(app.config['STATIC_FOLDER'],audio_file)
        engine.save_to_file(extracted_text, processed_image_path)
        engine.runAndWait()
        return render_template('result.html', extracted_text=extracted_text, processed_image=filename,audio_file=audio_file, text_class=text_class)

if __name__ == '__main__':
    app.run(debug=True)
