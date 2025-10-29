from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
model = joblib.load("model.joblib") if os.path.exists("model.joblib") else None

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, inputs={}, coefs=[], error=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", error="Model not trained yet. Run python train.py first.", result=None, inputs={}, coefs=[])

    try:
        # Read inputs
        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])
        assignments = float(request.form["assignments"])
        sleep = float(request.form["sleep"])
        previous = float(request.form["previous"])

        df = pd.DataFrame([[hours, attendance, assignments, sleep, previous]], 
                          columns=["HoursStudied", "AttendancePercent", "AssignmentsAvg", "SleepHours", "PreviousMarks"])

        prediction = model.predict(df)[0]
        prediction = round(prediction, 2)

        inputs = df.iloc[0].to_dict()
        coefs = list(zip(df.columns, model.coef_))

        return render_template("index.html", result=prediction, inputs=inputs, coefs=coefs, error=None)

    except Exception as e:
        print("Prediction error:", e)  # For debugging in console
        inputs = {
            "hours": request.form.get("hours", ""),
            "attendance": request.form.get("attendance", ""),
            "assignments": request.form.get("assignments", ""),
            "sleep": request.form.get("sleep", ""),
            "previous": request.form.get("previous", "")
        }
        return render_template("index.html", error="Please enter valid numbers.", result=None, inputs=inputs, coefs=[])
    
if __name__ == "__main__":
    app.run(debug=True)
