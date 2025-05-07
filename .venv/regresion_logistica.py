import os
import pyodbc
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cadena de conexión a la base de datos Somee
DB_CONN = (
    "DRIVER={SQL Server};"
    "SERVER=DataSetMachineLearning.mssql.somee.com;"
    "DATABASE=DataSetMachineLearning;"
    "UID=sgomez_SQLLogin_1;"
    "PWD=ph5nr42d1u;"
    "TrustServerCertificate=True;"
    "Persist Security Info=False;"
    "Packet Size=4096;"
)

def entrenar_modelo():
    try:
        print("Conexión a la base de datos...")
        conn = pyodbc.connect(DB_CONN)
        print("Conexión a la base de datos exitosa.")

        query = """
        SELECT Edad, IngresoMensual, NivelEducativo, EstadoCivil, TieneTarjetaCredito, AbriraCuenta
        FROM dbo.ClientesBanco
        """
        df = pd.read_sql(query, conn)
        conn.close()
        print("Consulta ejecutada correctamente.")
        print(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas.")

        # Limpieza de datos
        df.dropna(subset=["Edad", "IngresoMensual", "NivelEducativo", "EstadoCivil", "TieneTarjetaCredito", "AbriraCuenta"], inplace=True)

        # Normalizar y limpiar strings
        df["NivelEducativo"] = df["NivelEducativo"].str.strip().str.capitalize()
        df["EstadoCivil"] = df["EstadoCivil"].str.strip().str.capitalize()

        # Mapear variables categóricas a números
        df['NivelEducativo'] = df['NivelEducativo'].map({
            'Primaria': 1,
            'Secundaria': 2,
            'Universitario': 3,
            'Otro': 4
        })

        df['EstadoCivil'] = df['EstadoCivil'].map({
            'Soltero': 1,
            'Casado': 2,
            'Otro': 3
        })

        # Convertir TieneTarjetaCredito a binario si es texto como "Sí"/"No"
        df['TieneTarjetaCredito'] = df['TieneTarjetaCredito'].str.strip().str.lower().map({'sí': 1, 'si': 1, 'no': 0})
        
        # Eliminar filas que hayan quedado con NaN después de mapear
        df.dropna(inplace=True)

        # Variables independientes y dependiente
        X = df[["Edad", "IngresoMensual", "NivelEducativo", "EstadoCivil", "TieneTarjetaCredito"]]
        y = df["AbriraCuenta"]

        # Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Entrenar modelo
        model = LogisticRegression()
        model.fit(X_scaled, y)

        print("✅ Modelo entrenado correctamente.")
        return model, scaler

    except Exception as e:
        print(f"❌ Error en entrenar_modelo: {e}")
        return None, None



def predecir_apertura(entrada):
    model, scaler = entrenar_modelo()
    if model is None:
        return "Error al entrenar el modelo"

    # Convertimos los datos de entrada en DataFrame
    df_input = pd.DataFrame([[entrada["edad"], entrada["ingreso"], entrada["nivel"], entrada["estado"], entrada["tarjeta"]]],
                            columns=["Edad", "IngresoMensual", "NivelEducativo", "EstadoCivil", "TieneTarjetaCredito"])

    # Codificación de variables de entrada
    # Los valores ya vienen como números del formulario, no es necesario mapear
    df_input = df_input.astype(float)


    # Asegurarnos de que las columnas coinciden con las usadas para entrenar
    df_input = df_input.reindex(columns=["Edad", "IngresoMensual", "NivelEducativo", "EstadoCivil", "TieneTarjetaCredito"], fill_value=0)

    # Escalamos los datos de entrada
    df_scaled = scaler.transform(df_input)

    # Realizar predicción
    prediccion = model.predict(df_scaled)[0]
    return "Sí abrirá cuenta" if prediccion == 1 else "No abrirá cuenta"
