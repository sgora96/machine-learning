from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import the new linear regression analysis function
from linear_regression import train_linear_regression
from regresion_logistica import predecir_apertura
from connection_db import get_connection

app = Flask(__name__)

# Cargar el modelo si ya existe, de lo contrario, entrenarlo
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

# Redirigir raíz directamente al dashboard
@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("dashboard"))

@app.route("/iris", methods=["GET", "POST"])
def iris():
    prediction = None
    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            predicted_class = model.predict(features)[0]
            class_names = ["Setosa", "Versicolor", "Virginica"]
            prediction = class_names[predicted_class]
        except ValueError:
            prediction = "Error en los valores ingresados."

    return render_template("index.html", prediction=prediction)

@app.route("/regresion-lineal", methods=['GET', 'POST'])
def linear_regression():
    results = None
    error = None
    
    if request.method == 'POST':
        try:
            results = train_linear_regression()  
            if results is None:
                error = "No se pudo procesar los datos de la base de datos."
        except Exception as e:
            error = f"Ocurrió un error: {str(e)}"
    
    return render_template("regression.html", results=results, error=error)

@app.route('/logistica', methods=['GET', 'POST'])
def logistica():
    if request.method == 'POST':
        try:
            edad = float(request.form['edad'])
            ingresos = float(request.form['ingresos'])
            nivel = int(request.form['nivel'])
            estado = int(request.form['estado'])
            tarjeta = int(request.form['tarjeta'])

            datos = [[edad, ingresos, nivel, estado, tarjeta]]
            resultado = predecir_apertura({
                "edad": edad,
                "ingreso": ingresos,
                "nivel": nivel,
                "estado": estado,
                "tarjeta": tarjeta
            })

            return render_template('logistic.html', resultado=resultado)

        except Exception as e:
            return f"Error al procesar el formulario: {e}", 400

    return render_template('logistic.html')

@app.route('/menu-modelos')
def mainModel():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Id, Nombre FROM ModelosMLClasificacion")
    modelos = cursor.fetchall()
    conn.close()
    return render_template('menu_modelos.html', modelos=modelos)

@app.route('/modelo/<int:modelo_id>')
def modelo(modelo_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Nombre, Descripcion, Fuente, ContenidoGrafico FROM ModelosMLClasificacion WHERE Id = ?", modelo_id)
    modelo = cursor.fetchone()
    conn.close()
    if modelo:
        return render_template('modelo.html', modelo=modelo)
    else:
        return "Modelo no encontrado", 404

@app.route("/mapa-mental")
def mapa_mental():
    return render_template("mapa_mental.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)
