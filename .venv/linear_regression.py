import os
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Configuración de conexión a la base de datos
DB_CONN = "DRIVER={SQL Server};SERVER=localhost\\SQLEXPRESS;DATABASE=datasheet_act4;Trusted_Connection=yes;"

def train_linear_regression():
    try:
        # Conectar a la base de datos
        conn = pyodbc.connect(DB_CONN)
        query = "SELECT superficie_cultivada, produccion FROM dbo.datasheet"
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            print("⚠️ No se encontraron datos en la tabla.")
            return None

        df.dropna(inplace=True)

        # Definir variables
        X = df[['superficie_cultivada']].values  
        y = df['produccion'].values              

        # Entrenar modelo de regresión lineal
        model = LinearRegression()
        model.fit(X, y)

        coef = model.coef_[0]
        intercept = model.intercept_

        y_pred = model.predict(X)

        # 📌 Asegurar que la carpeta 'static/' está dentro del directorio del script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        static_folder = os.path.join(base_dir, "static")
        if not os.path.exists(static_folder):
            os.makedirs(static_folder)

        # Guardar la imagen correctamente
        image_path = os.path.join(static_folder, "regression_plot.png")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df['superficie_cultivada'], y=df['produccion'], color='blue', label="Datos reales")
        plt.plot(df['superficie_cultivada'], y_pred, color='red', linewidth=2, label="Regresión Lineal")
        plt.xlabel("Superficie Cultivada")
        plt.ylabel("Producción")
        plt.title("Regresión Lineal: Superficie Cultivada vs Producción")
        plt.legend()
        plt.grid()
        
        plt.savefig(image_path)
        plt.close()

        # Devolver ruta relativa para que Flask pueda encontrar la imagen
        return {
            "coeficiente": coef,
            "interseccion": intercept,
            "image_path": os.path.relpath(image_path, base_dir).replace("\\", "/")
        }

    except Exception as e:
        print(f"Error en train_linear_regression: {e}")
        return {"error": str(e)}