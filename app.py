from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model=joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    try:
        hours = float(request.form['hours'])
        prev_score = float(request.form['score'])
        attendance = float(request.form['attendance'])
        assignments = float(request.form['assignments'])
        internet = float(request.form['internet'])
        sleep = float(request.form['sleep'])

        input_data = np.array([[hours, prev_score, attendance, assignments, internet, sleep]])
        result = model.predict(input_data)

        output = "Pass ✅" if result[0] == 1 else "Fail ❌"
        return render_template('index.html', prediction_text=output)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")
    
if __name__ == '__main__':
    app.run(debug=True)