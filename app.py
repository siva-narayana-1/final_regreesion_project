from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
with open("slr_model.pkl", "rb") as f:
    slr_model = pickle.load(f)

with open("mlr_model.pkl", "rb") as f:
    mlr_model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/slr", methods=["GET", "POST"])
def slr():
    prediction = None
    if request.method == "POST":
        try:
            x_val = float(request.form["x_value"])
            prediction = slr_model.predict(np.array([[x_val]]))[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("slr.html", prediction=round(prediction))


@app.route("/mlr", methods=["GET", "POST"])
def mlr():
    prediction = None
    if request.method == "POST":
        try:
            rd = float(request.form["rd"])
            admin = float(request.form["admin"])
            marketing = float(request.form["marketing"])
            state = request.form["state"]

            # map state to numeric
            state_map = {"New York": 1, "California": 2, "Florida": 3}
            state_val = state_map.get(state, 0)

            features = np.array([[rd, admin, marketing, state_val]])
            prediction = mlr_model.predict(features)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("mlr.html", prediction= round(prediction))



if __name__ == "__main__":
    app.run(debug=True)
