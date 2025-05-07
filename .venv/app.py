from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Import the new linear regression analysis function
from linear_regression import train_linear_regression
from regresion_logistica import predecir_apertura

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


@app.route("/regresion-lineal", methods=['GET', 'POST'])
def linear_regression():
    results = None
    error = None
    
    if request.method == 'POST':
        try:
            # Llamar a la función sin argumentos
            results = train_linear_regression()  
            
            if results is None:
                error = "No se pudo procesar los datos de la base de datos."
        
        except Exception as e:
            error = f"Ocurrió un error: {str(e)}"
    
    return render_template("regression.html", 
                            results=results, 
                            error=error)
    
@app.route('/logistica', methods=['GET', 'POST'])
def logistica():
    if request.method == 'POST':
        try:
            edad = float(request.form['edad'])
            ingresos = float(request.form['ingresos'])
            nivel = int(request.form['nivel'])
            estado = int(request.form['estado'])
            tarjeta = int(request.form['tarjeta'])

            # Aquí pasa los datos a tu modelo
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


if __name__ == "__main__":
    app.run(debug=True)
