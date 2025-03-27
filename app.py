from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Cargar el modelo si ya existe, de lo contrario, entrenarlo
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    # Cargar dataset Iris (ejemplo supervisado)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Guardar modelo
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])
            
            # Convertir los valores en un array numpy
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Predecir la clase de la flor
            predicted_class = model.predict(features)[0]
            class_names = ["Setosa", "Versicolor", "Virginica"]
            prediction = class_names[predicted_class]
        except ValueError:
            prediction = "Error en los valores ingresados."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    
@app.route("/mapa-mental")
def mapa_mental():
    return render_template("mapa_mental.html")

