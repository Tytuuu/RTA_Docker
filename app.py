from flask import Flask
from flask import request
from flask import jsonify
import pickle
import perceptron
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Docker!'

@app.route('/api/predict', methods = ["GET"])
def get_prediction():
    sepal_length = float(request.args.get('sepal length (cm)'))
    petal_length = float(request.args.get('petal length (cm)'))
    features = [sepal_length, petal_length]
    print(features)
    
    with open("iris.pkl", "rb") as picklefile:
        model = pickle.load(picklefile)
    print(model)
   
    predicted_class = int(model.predict(features))
   
    return jsonify(features = features, predicted_class = predicted_class)