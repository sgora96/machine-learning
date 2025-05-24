import pyodbc

def get_connection():
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=DataSetMachineLearning.mssql.somee.com;'
        'DATABASE=DataSetMachineLearning;'
        'UID=sgomez_SQLLogin_1;'
        'PWD=ph5nr42d1u;'
        'Persist Security Info=False;'
        'TrustServerCertificate=Yes;'
        'Workstation ID=DataSetMachineLearning.mssql.somee.com;'
        'Packet Size=4096;'
    )
    return pyodbc.connect(conn_str)
