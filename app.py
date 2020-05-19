import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pprint import pprint
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()


app = Flask(__name__)
model = pickle.load(open('test.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    # data = [int_features[0:4], int_features[4:8], int_features[8:12],int_features[12:16]]
    data = []
    j = 0
    for i in range(len(int_features)//4):
        data.append(int_features[j:j+4])
        j = j+4
    # print(data)
    # data = [int_features[i:i+4] for i in range(len(int_features))]
    x = np.array(data)
    print(pd.DataFrame(data))
    x[:, 3] = labelencoder.fit_transform(x[:, 3])
    x = x.astype(np.float64)
    ct = ColumnTransformer([("", OneHotEncoder(), [3])], remainder='passthrough')

    x = ct.fit_transform(x)
    x = x[:, 1:]
    output = model.predict(x)
    return render_template('index.html', prediction_text=".\n".join([f'Profit for %s is %d' for i in range(len(data))]) % (int_features[3],round(output[0],2),int_features[7],round(output[1],2),int_features[11],round(output[2],2),int_features[15],round(output[3],2)))

if __name__ == "__main__":
    app.run(debug=True)
