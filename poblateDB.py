
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://juancaniguante8:juanpablo1997@cluster0.prilz3h.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client['test']

#Cargar archivo csv
import pandas as pd


archivos = ['data/categorieMLC1672.csv','data/categorieMLC5068.csv', 'data/categorieMLC9240.csv', 'data/categorieMLC48906.csv', 'data/usuariosMLC1672.csv', 'data/usuariosMLC5068.csv', 'data/usuariosMLC9240.csv', 'data/usuariosMLC48906.csv']

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    #Agregar categoria 1672 Discos Duros y Solidos
    dfCategorie = pd.read_csv(archivos[0])
    #Convert
    data = dfCategorie.to_dict('records')
    #Insertar
    db['categoria1672'].insert_many(data)
    #Usuarios 
    dfUsuarios = pd.read_csv(archivos[4])
    #Convert
    data = dfUsuarios.to_dict('records')
    #Insertar
    db['usuarios1672'].insert_many(data)
    #Agregar categoria 5068
    dfCategorie = pd.read_csv(archivos[1])
    #Convert
    data = dfCategorie.to_dict('records')
    #Insertar
    db['categoria5068'].insert_many(data)
    #Usuarios
    dfUsuarios = pd.read_csv(archivos[5])
    #Convert
    data = dfUsuarios.to_dict('records')
    #Insertar
    db['usuarios5068'].insert_many(data)
    #Agregar categoria 9240
    dfCategorie = pd.read_csv(archivos[2])
    #Convert
    data = dfCategorie.to_dict('records')
    #Insertar
    db['categoria9240'].insert_many(data)
    #Usuarios
    dfUsuarios = pd.read_csv(archivos[6])
    #Convert
    data = dfUsuarios.to_dict('records')
    #Insertar
    db['usuarios9240'].insert_many(data)
    #Agregar categoria 48906
    dfCategorie = pd.read_csv(archivos[3])
    #Convert
    data = dfCategorie.to_dict('records')
    #Insertar
    db['categoria48906'].insert_many(data)
    #Usuarios
    dfUsuarios = pd.read_csv(archivos[7])
    #Convert
    data = dfUsuarios.to_dict('records')
    #Insertar
    db['usuarios48906'].insert_many(data)
    #Cerrar la conexion
    client.close()
except Exception as e:
    print(e)