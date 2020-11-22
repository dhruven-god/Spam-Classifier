from flask import Flask, render_template, url_for,request
import pickle
import numpy

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

cv = pickle.load(open('transform.pkl','rb'))

@app.route('/')

def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():

    text = request.form['text']
    data = [text]
    data_vector = cv.transform(data).toarray()
    prediction = model.predict(data_vector)

    return render_template('result.html', prediction=prediction)



if __name__ == "__main__":
    app.run(debug=True)