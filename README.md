# EchoPressVision - OpenCV & OCR 
![Screenshot (181)](https://github.com/arinsharma123/Cantilever-OCR-/assets/128144029/db831cf5-b94d-4eee-a10e-95108aadf248)

## Overview:
EchoPressVision is designed to assist individuals with visual impairments by enabling them to grasp the contents of newspapers through real-time audio conversion. The project integrates three major subprocesses: a text classification model, OpenCV & OCR, and a web application to deliver a seamless user experience.
#### Tech stack: cv2, pytesseract, sklearn, pyttsx3, textblob, nltk, Flask, Jupyter lab, VSCode. 

## Pipeline:

### 1. Text Classification Model:
* Random Forest Classifier trained in a large 3rd party dataset to achieve 89% text classification accuracy.
* Raw data had 4 classes -> 1-World, 2-Sports, 3-Business, 4-Sci/Tech. After a thorough cleaning, data is vectorized using Bag of Words embeddings.
* TECH STACK: sklearn, nltk, re, numpy, pandas.
![Screenshot (188)](https://github.com/arinsharma123/Cantilever-OCR-/assets/128144029/0b2604e5-0bd5-4af6-a767-f5033bf49dd9)

### 2. OpenCV & OCR:
*Newspaper images are inputted, and computer vision techniques are employed to identify text-centric regions, highlighted by bounding boxes.
*Text is extracted from these regions, undergoes basic preprocessing and an audio file is subsequently generated.
* TECH STACK: pytesseract, cv2, pyttsx3, matplotlib, spellchecker, textblob, nltk, re.

![Untitled design (1)](https://github.com/arinsharma123/Cantilever-OCR-/assets/128144029/04258b28-33b2-440f-84e9-4fee42d3c867)

![Untitled design](https://github.com/arinsharma123/Cantilever-OCR-/assets/128144029/8f8a7c77-6aca-4a7a-8eb5-b1816fd64366)

### 3. Web Application:
* Necessary functions and models are imported for the sequential processing of the web application.
* The uploads and audio files are saved in the console in real time and then fetched to present on the local host.
* TECH STACK: Flask, werkzeug, pytesseract, cv2, re, textblob, pyttsx3,nltk
```python
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
```

## Implementation:

User Interface: A clean, minimalistic design with an upload box for newspaper images is the home and on submission of the image, the user is directed to the results page where a display section for the extracted text and audio playback is present.
Real-Time Processing: Efficiently processes and converts the uploaded images to text and subsequently to audio in real-time.
![Screenshot (180)](https://github.com/arinsharma123/Cantilever-OCR-/assets/128144029/30584314-9c58-4f60-aed5-a850441702d0)
