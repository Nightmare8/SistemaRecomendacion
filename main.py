# Description: This file contains the main code of the API
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from pydantic import BaseModel
import os
#En este codigo se aplicara el metodo de filtrado colaborativo basado en contenido
import pandas as pd
import json
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import SnowballStemmer

#Conection to mongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://juancaniguante8:juanpablo1997@cluster0.prilz3h.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['test']

app = FastAPI()

#Definir puerto de la aplicacion
# port = int(os.environ.get("PORT", 8000))
#Definiciones de columnas
stopword_es = nltk.corpus.stopwords.words('spanish')
text_columns = ['descripcion', 'titulo', 'tags', 'ciudad', 'opiniones', 'nombre', 'tagsVendedor', 'regionVendedor', 'ciudadVendedor']
categorical_columns = ['region']
numeric_columns = ['precio', 'cantidadReviews', 'unaEstrella', 'dosEstrellas', 'tresEstrellas', 'cuatroEstrellas', 'cincoEstrellas', 'reviewPromedio', 'likes', 'dislikes', 'reputacion', 'reputacionEstrellas', 'completadas','canceladas', 'total', 'ratingPositivo', 'ratingNeutral', 'ratingNegativo', 'ventas', 'reclamos', 'entregasRetrasadas', 'cancelaciones']

stemmer = SnowballStemmer('spanish')
stopword_es = nltk.corpus.stopwords.words('spanish')

def clean (texto):
    texto = str(texto)
    if (texto == 'nan'):
        return texto
    if pd.isna(texto):
        return ""
    if (texto == ''):
        return texto
    texto = str(texto)
    if texto == 'nan':
        return ""
    texto = texto.lower()
    texto = texto.replace('[^\w\s]','')
    #Replace special characters
    texto = texto.strip()
    texto = texto.replace('\d+','')
    texto = texto.replace('\n+','')
    texto = texto.replace('\r+','')
    texto = texto.replace('\t+','')
    return texto

def clean_and_join_attributes (row, columns):
    cleaned_row = []
    for item in row[columns]:
        if pd.isna(item):
            cleaned_item = ""
        else:
            cleaned_item = clean(str(item))
        cleaned_row.append(cleaned_item)
    return " ".join(cleaned_row)


def stem_tokens(tokens, stemmer):
    stemmed = [stemmer.stem(item) for item in tokens]
    return stemmed
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if (re.search('[a-zA-Z]', token)):
            filtered_tokens.append(token)
    #Exclude stop words from stemmed words
    stems = [stemmer.stem(t) for t in filtered_tokens if t not in stopword_es]

    return stems

def process_attribute_column(products, userProducts, attribute_columns, n_components, stemming):
    products['all_attributes'] = products[attribute_columns].apply(lambda x: clean_and_join_attributes(x, attribute_columns), axis=1)
    userProducts['all_attributes'] = userProducts[attribute_columns].apply(lambda x: clean_and_join_attributes(x, attribute_columns), axis=1)
    combined_data = pd.concat([products['all_attributes'], userProducts['all_attributes']], ignore_index=True)
    #Aplicar TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words = stopword_es)
    if (stemming):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)
    tfidf_data = tfidf_vectorizer.fit_transform(combined_data.values.astype('U'))

    #Aplicar SVD
    svd = TruncatedSVD(n_components=n_components)
    reduced_data = svd.fit_transform(tfidf_data)

    #Dividir los datos reducidos en productos y usuarios
    reduced_productos = reduced_data[:products.shape[0]]
    reduced_users = reduced_data[products.shape[0]:]
    return reduced_productos, reduced_users

def reccomendationAllColumns(productos,userProducts, text_columns, numeric_columns,atributes_columns, n_components=2, keyWords = None, stemming=False):
    print (productos)
    print (userProducts)
    #aplicar el vectorizador a los productos
    tfidf_vectorizer = TfidfVectorizer(stop_words = stopword_es)
    if (stemming):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)

    #Clean text of the product
    productosCopy = productos.copy()
    productosCopy['combined_text'] = productos[text_columns].apply(lambda x: clean_and_join_attributes(x, text_columns), axis=1)
    
    #Clean Text of user
    userProductsCopy = userProducts.copy()
    userProductsCopy['combined_text'] = userProducts[text_columns].apply(lambda x: clean_and_join_attributes(x, text_columns), axis=1)
    combined_text = pd.concat([productosCopy['combined_text'], userProductsCopy['combined_text']], ignore_index=True)
    
    #Aplicar TF-IDF a textos
    tfidf_vectorizer.fit(combined_text.values.astype('U'))
    #Solamente se unen las columnas de texto de cada fila
    text_features_products = tfidf_vectorizer.transform(productosCopy['combined_text'].values.astype('U'))
    text_features_users = tfidf_vectorizer.transform(userProductsCopy['combined_text'].values.astype('U'))
    if (keyWords != None):
        value_new_word = 10
        keyword_weights = dict()
        for word in keyWords:
            keyword_weights[word] = value_new_word
            keyword_weights[word.lower()] = value_new_word
            keyword_weights[word.upper()] = value_new_word
        
        additional_weights = np.array([keyword_weights.get(word, 1) for word in tfidf_vectorizer.get_feature_names_out()])
        
        text_features_products = text_features_products.multiply(additional_weights)
        text_features_users = text_features_users.multiply(additional_weights)
    
    #Obtener similitud de texto
    text_similarity = cosine_similarity(text_features_users, text_features_products)
    #Explorar la similitud de atributos
    if (n_components != 0):
        reduced_productos, reduced_users = process_attribute_column(productosCopy, userProductsCopy, atributes_columns, n_components, stemming)
        #Calcular similitud
        attribute_similarity = cosine_similarity(reduced_users, reduced_productos)
    else:
        #Calcular la similitud de atributos sin SVD
        tfidf_vectorizer_attributes = TfidfVectorizer(stop_words = stopword_es)
        if (stemming):
            tfidf_vectorizer_attributes = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)
            
        #Limpieza de atributos
        productosCopy['combined_attributes'] = productos[atributes_columns].apply(lambda x: clean_and_join_attributes(x, atributes_columns), axis=1)
        userProductsCopy['combined_attributes'] = userProducts[atributes_columns].apply(lambda x: clean_and_join_attributes(x,atributes_columns), axis=1)
        combined_attributes = pd.concat([productosCopy['combined_attributes'], userProductsCopy['combined_attributes']], ignore_index=True)
        
        #Aplicar TF-IDF a atributos
        tfidf_vectorizer_attributes.fit(combined_attributes.values.astype('U'))
        text_features_products_attributes = tfidf_vectorizer_attributes.transform(productosCopy['combined_attributes'].values.astype('U'))
        text_features_users_attributes = tfidf_vectorizer_attributes.transform(userProductsCopy['combined_attributes'].values.astype('U'))
        attribute_similarity = cosine_similarity(text_features_users_attributes, text_features_products_attributes)
    
    #Calcular similitud numerica
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(productos[numeric_columns])
    numeric_features_sparse = sp.csr_matrix(numeric_features)
    #print (numeric_features_sparse.shape)
    #User features
    user_numeric_features = scaler.transform(userProducts[numeric_columns])
    user_numeric_features_sparse = sp.csr_matrix(user_numeric_features)
    #print (user_numeric_features_sparse.shape)
    #Obtain similarity
    numeric_similarity = cosine_similarity(user_numeric_features_sparse, numeric_features_sparse)
    return (text_similarity, numeric_similarity, attribute_similarity)


def get_reccomendation (productos, userProducts, text_columns, numeric_columns,atributes_columns, n_components,  keyWords=None, stemming=False, all_columns=True ):
    
    if (all_columns):
        text_similarity, numeric_similarity, attribute_similarity = reccomendationAllColumns(productos,userProducts, text_columns, numeric_columns, atributes_columns, n_components, keyWords, stemming)

        #Aplicar magnitud y suma de similitudes
        total_columns = len(text_columns) + len(numeric_columns) + len(atributes_columns)
        text_weight = (len(text_columns) + len(atributes_columns))/total_columns
        numeric_weight = len(numeric_columns)/total_columns
        attribute_weight = len(atributes_columns)/total_columns
        
        #Similitud combinada
        similitud_combinada = text_weight * text_similarity + numeric_weight * numeric_similarity + attribute_weight * attribute_similarity
        
        return  similitud_combinada
    
def get_product(idProducto, productos):
    #Obtener la informacion del producto
    producto = productos[productos['idProducto'] == idProducto].iloc[0].values.tolist()
    print ("tiro error aqui")
    print ("producto", producto)
    #Obtain the columns
    response = dict()
    for col in productos.columns:
        value = producto[productos.columns.get_loc(col)]
        if (type(value) == np.float64):
            value = float(value)
        #Replace nan values with empty string
        if (type(value) == float and np.isnan(value)):
            value = ""
        response[col] = value
    return response

#Devolver los datos de un producto, recibiendo el id del producto
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@app.get("/product/{idProducto}/{categoria}")
async def read_product(idProducto: str, categoria: str):    
    #Se cargan los datos de los productos
    productos = db['categoria' + categoria].find()
    #Convert to dataframe
    productos = pd.DataFrame(list(productos))
    #Delete the object id
    productos = productos.drop('_id', axis=1)
    # productos = pd.read_csv('data/categorie'+ categoria + '.csv', sep=',')
    response = get_product(idProducto, productos)
    #Transform dict to json
    response = json.dumps(response, cls=NpEncoder)
    #response = JSONResponse(content=response)
    return response

#Este endpoint sera para obtener todos los productos de una categoria
@app.get("/categorie/{categoria}")
async def read_categorie(categoria: str):
    #Se cargan los datos de los productos
    try:
        productos = pd.read_csv('data/categorie'+ categoria + '.csv', sep=',')
        response = productos.to_json(orient="records")
        response = JSONResponse(content=response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Verificar el catalogo y obtener los productos que no estan en el catalogo
@app.get("/catalogue/{categoria}")
async def read_catalogue(categoria: str):
    try:
        #Se cargan los datos de los productos
        productos = pd.read_csv('data/categorie'+ categoria + '.csv', sep=',')
        #Se cargan los datos de los usuarios
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class NewItem(BaseModel):
    idProducto: str
    categoria: str

@app.post('/addPurchaseOrder')
async def add_register(item: NewItem):
    try:
        print ("entra a la funcion")
        idProducto = item.idProducto
        categoria = item.categoria
        print ("idProducto", idProducto)
        print ("categoria", categoria)
        #Obtener el producto
        collection = db['categoria' + categoria]

        #Buscar el producto
        producto = collection.find_one({'idProducto': idProducto})

        if (producto == None):
            raise HTTPException(status_code=500, detail="Producto no encontrado")
        
        #Si no existe el producto, agregarlo a la coleccion de usuario
        #Obtener la coleccion de usuarios
        userCollection = db['usuarios' + categoria]
        #Dropear _id de producto
        producto.pop('_id')
        userCollection.insert_one(producto)

        return "success"
    except Exception as e:
        print ("error", e)
        raise HTTPException(status_code=500, detail=str(e))

#Change to post
class Item(BaseModel):
    categoria: str or None
    cantidad: int
    keyWords: list = None
    stemming: bool = False
    n_components: int = 0

@app.post("/recommendations")
async def read_root(item: Item):
    #Se cargan los datos de los usuarios
    try:
        print ("item", item)
        categoria = item.categoria #*SIN EL MLC
        cantidad = item.cantidad
        keyWords = item.keyWords
        stemming = item.stemming
        n_components = item.n_components
        #Usar los datos para cargar segun la categoria
        #Se cargan los datos de los usuarios desde mongo
        userProducts = db['usuarios' + categoria].find()
        #Convertir a dataframe
        userProducts = pd.DataFrame(list(userProducts))
        #Filtrar los productos de los usuarios ocupando la categoria de interes
        #Se cargan los datos de los productos
        #productos = pd.read_csv('data/categorie'+ categoria +'.csv', sep=',')
        productos = db['categoria' + categoria].find()
        #Convertir a dataframe
        productos = pd.DataFrame(list(productos))
        with open('data/categoriesRegistersComplete.json', 'r') as archivo:
            data = json.load(archivo)
        datosCategoria = data['MLC'+categoria]
        atributes_columns = []
        for atributo in datosCategoria['attributes']:
            atributes_columns.append(atributo['id'])

        cantidad = cantidad + 1
        similitud_combinada = get_reccomendation(productos,userProducts, text_columns, numeric_columns, atributes_columns,  n_components, keyWords,stemming  )
        #Aplicar magnitud y suma de similitudes
        magnitudes = np.linalg.norm(similitud_combinada, axis=1)
        sum_similitudes = similitud_combinada.sum(axis=0) / magnitudes.sum()
        top_indices = sum_similitudes.argsort()[:-cantidad:-1]
        listaRespuesta = list()
        for i in top_indices:
            #Transform to json
            #Podria agregar toda la info del producto
            #Transformar a dict
            
            listaRespuesta.append({
                "idProducto": productos['idProducto'].iloc[i],
                "similitud": sum_similitudes[i],
                "categoria": categoria
            })
        response = {
            "categoria": categoria,
            "cantidad": cantidad-1,
            "productos": listaRespuesta
        }
        response = JSONResponse(content=response)
        return response
    
    except Exception as e:
        print ("error", e)
        raise HTTPException(status_code=500, detail=str(e))
