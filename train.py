# train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the data
df = pd.read_csv("data/marks.csv")  # Use your actual CSV filename

# Features and target
X = df[["HoursStudied", "AttendancePercent", "AssignmentsAvg", "SleepHours", "PreviousMarks"]]
y = df["Marks"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "model.joblib")

print("Model trained and saved as model.joblib")
