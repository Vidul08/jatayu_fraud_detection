import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template

# Load data for app1
data_app1 = pd.read_csv('sample_data.csv')
scaler_app1 = StandardScaler()
X_app1 = scaler_app1.fit_transform(data_app1.drop('fraud', axis=1))
y_app1 = data_app1['fraud']
model_app1 = LogisticRegression()
model_app1.fit(X_app1, y_app1)

# Load data for app2
data_app2 = pd.read_csv('healthcare_data.csv')
scaler_app2 = StandardScaler()
X_app2 = scaler_app2.fit_transform(data_app2.drop('fraudulent', axis=1))
y_app2 = data_app2['fraudulent']
model_app2 = LogisticRegression()
model_app2.fit(X_app2, y_app2)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/app1', methods=['GET', 'POST'])
def app1():
    if request.method == 'POST':
        # Get the input data from the user
        input_data = request.form.to_dict()

        # Preprocess the input data
        X = scaler_app1.transform(pd.DataFrame(input_data, index=[0]))

        # Make a prediction using the model
        prediction = model_app1.predict(X)[0]

        # Render a template with the prediction result
        return render_template('result.html', prediction=bool(prediction))

    return render_template('index.html')

@app.route('/app2', methods=['GET', 'POST'])
def app2():
    if request.method == 'POST':
        # Get the input data from the user
        input_data = request.form.to_dict()

        # Preprocess the input data
        X = scaler_app2.transform(pd.DataFrame(input_data, index=[0]))

        # Make a prediction using the model
        prediction = model_app2.predict(X)[0]

        # Check if the rectification is fraudulent or not
        if prediction == 1:
            result = "Fraudulent rectification"
        else:
            result = "Legitimate rectification"

        # Render a template with the prediction result
        return render_template('result1.html', prediction=result)

    return render_template('index1.html')

@app.route('/app1/predict', methods=['POST'])
def app1_predict():
    # Get the input data from the user
    input_data = request.form.to_dict()

    # Preprocess the input data
    X = scaler_app1.transform(pd.DataFrame(input_data, index=[0]))

    # Make a prediction using the model
    prediction = model_app1.predict(X)[0]

    # Render a template with the prediction result
    return render_template('result.html', prediction=bool(prediction))

@app.route('/app2/predict', methods=['POST'])
def app2_predict():
    # Get the input data from the user
    input_data = request.form.to_dict()

    # Preprocess the input data
    X = scaler_app2.transform(pd.DataFrame(input_data, index=[0]))

    # Make a prediction using the model
    prediction = model_app2.predict(X)[0]

    # Check if the rectification is fraudulent or not
    if prediction == 1:
        result = "Fraudulent rectification"
    else:
        result = "Legitimate rectification"

    # Render a template with the prediction result
    return render_template('result1.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
