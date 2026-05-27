from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# load the trained model
model = pickle.load(open('model.pkl', 'rb'))

#home page
@app.route('/')
def home():
    return render_template('index.html')

#prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # get form values
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])


    # convert into numpy array
    features = np.array([[sepal_length,sepal_width, petal_length, petal_width]])

    # predict the class 
    prediction = model.predict(features)

    # flowers classes
    flowerclasses = ['setosa', 'versicolor', 'virginica']

    result = flowerclasses[prediction[0]]

    return render_template('index.html', prediction_text = f'Predicted Iris Flower Class: {result}')

#run the app
if __name__ == '__main__':
    app.run(debug=True)