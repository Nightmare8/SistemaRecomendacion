# Description: This file contains the main code of the API
from typing import Union
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from pydantic import BaseModel
#En este codigo se aplicara el metodo de filtrado colaborativo basado en contenido
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import SnowballStemmer

#Todo esto sera ocupando solo una categoria por el momento

app = FastAPI()

stopword_es = nltk.corpus.stopwords.words('spanish')
text_columns = ['descripcion', 'titulo', 'tags', 'ciudad', 'opiniones', 'nombre', 'tagsVendedor', 'regionVendedor', 'ciudadVendedor']
categorical_columns = ['region']
numeric_columns = ['precio', 'cantidadReviews', 'unaEstrella', 'dosEstrellas', 'tresEstrellas', 'cuatroEstrellas', 'cincoEstrellas', 'reviewPromedio', 'likes', 'dislikes', 'reputacion', 'reputacionEstrellas', 'completadas','canceladas', 'total', 'ratingPositivo', 'ratingNeutral', 'ratingNegativo', 'ventas', 'reclamos', 'entregasRetrasadas', 'cancelaciones']

stemmer = SnowballStemmer('spanish')
stopword_es = nltk.corpus.stopwords.words('spanish')

#vectorizer = TfidfVectorizer(stop_words = stopword_es)

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

def process_attribute_column(products, userProducts, attribute_columns, n_components):
    products['all_attributes'] = products[attribute_columns].apply(lambda x: clean_and_join_attributes(x, attribute_columns), axis=1)
    userProducts['all_attributes'] = userProducts[attribute_columns].apply(lambda x: clean_and_join_attributes(x, attribute_columns), axis=1)
    combined_data = pd.concat([products['all_attributes'], userProducts['all_attributes']], ignore_index=True)
    #Aplicar TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words = stopword_es)
    tfidf_data = tfidf_vectorizer.fit_transform(combined_data.values.astype('U'))

    #Aplicar SVD
    svd = TruncatedSVD(n_components=n_components)
    reduced_data = svd.fit_transform(tfidf_data)
    #Imprimir la varianza
    # print (reduced_data)
    #Dividir los datos reducidos en productos y usuarios
    reduced_productos = reduced_data[:products.shape[0]]
    reduced_users = reduced_data[products.shape[0]:]
    return reduced_productos, reduced_users

def reccomendationAllColumns(productos,userProducts, text_columns, numeric_columns,atributes_columns, n_components=2, keyWords = None, stemming=False):
    #aplicar el vectorizador a los productos
    tfidf_vectorizer = TfidfVectorizer(stop_words = stopword_es)
    tfidf_vectorizer_attributes = TfidfVectorizer(stop_words = stopword_es)
    if (stemming):
        tfidf_vectorizer = TfidfVectorizer(stop_words = stopword_es, tokenizer=tokenize)
        tfidf_vectorizer_attributes = TfidfVectorizer(stop_words = stopword_es, tokenizer=tokenize)
    #clearn text of products
    productosCopy = productos.copy()
    productosCopy = productosCopy.replace(np.nan, '', regex=True)
    productosCopy[text_columns] = productosCopy[text_columns].astype(str)
    productosCopy[text_columns] = productosCopy[text_columns].apply(clean)
    productosCopy[atributes_columns] = productosCopy[atributes_columns].astype(str)
    productosCopy[atributes_columns] = productosCopy[atributes_columns].apply(clean)
    #Clean Text of user
    userProductsCopy = userProducts.copy()
    userProductsCopy = userProductsCopy.replace(np.nan, '', regex=True)
    userProductsCopy[text_columns] = userProductsCopy[text_columns].astype(str)
    userProductsCopy[text_columns] = userProductsCopy[text_columns].apply(clean)
    userProductsCopy[atributes_columns] = userProductsCopy[atributes_columns].astype(str)
    userProductsCopy[atributes_columns] = userProductsCopy[atributes_columns].apply(clean)
    #Combinar todo los textos de todas las columnas
    combinedProducts = " ".join(productosCopy[text_columns].sum())
    combinedUser = " ".join(userProductsCopy[text_columns].sum())
    combinedProductsAtributes = " ".join(productosCopy[atributes_columns].sum())
    combinedUserAtributes = " ".join(userProductsCopy[atributes_columns].sum())
    #Ajustamos el vectorizador con todas las palabras
    tfidf_vectorizer.fit([combinedProducts, combinedUser])
    tfidf_vectorizer_attributes.fit([combinedProductsAtributes, combinedUserAtributes])
    productosCopy['combined_text'] = productos.apply(lambda x: clean_and_join_attributes(x, text_columns), axis=1)
    text_features_products = tfidf_vectorizer.transform(productosCopy['combined_text'].values.astype('U'))
    userProductsCopy['combined_text'] = userProducts.apply(lambda x: clean_and_join_attributes(x, text_columns), axis=1)
    text_features_users = tfidf_vectorizer.transform(userProductsCopy['combined_text'].values.astype('U'))
   
    #Calcular atributos
    
    if (keyWords != None):
        keyword_weights = {word: 10 for word in keyWords}
        #keyWords.append(tfidf_vectorizer.get_feature_names_out()[0])
        additional_weights = np.array([keyword_weights.get(word, 1) for word in tfidf_vectorizer.get_feature_names_out()])
        text_features_products = text_features_products.multiply(additional_weights)
        text_features_users = text_features_users.multiply(additional_weights)
    #Aplicar SVD
    text_similarity = cosine_similarity(text_features_users, text_features_products)
    if (n_components != 0):
        reduced_productos, reduced_users = process_attribute_column(productosCopy, userProductsCopy, atributes_columns, n_components)
        #Calcular similitud
        attribute_similarity = cosine_similarity(reduced_users, reduced_productos)
        print ("dimension de similitud de atributos", attribute_similarity.shape)
        #Segun cantidad de componentes, se aplica un peso
    else:
        productosCopy['combined_attributes'] = productos.apply(lambda x: clean_and_join_attributes(x, atributes_columns), axis=1)
        text_features_products_attributes = tfidf_vectorizer_attributes.transform(productosCopy['combined_attributes'].values.astype('U'))
        userProductsCopy['combined_attributes'] = userProducts.apply(lambda x: clean_and_join_attributes(x,atributes_columns), axis=1)
        text_features_users_attributes = tfidf_vectorizer_attributes.transform(userProductsCopy['combined_attributes'].values.astype('U'))
        attribute_similarity = cosine_similarity(text_features_users_attributes, text_features_products_attributes)
    #Numeric features
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
    return (text_similarity, numeric_similarity, attribute_similarity, text_features_users, text_features_products, user_numeric_features_sparse, numeric_features_sparse, tfidf_vectorizer, tfidf_vectorizer_attributes)


def get_reccomendation (productos, userProducts, text_columns, numeric_columns,atributes_columns, n_components,  keyWords=[], stemming=False, all_columns=True ):
    
    if (all_columns):
        text_similarity, numeric_similarity, attribute_similarity, text_features_users, text_features_products, user_numeric_features_sparse, numeric_features_sparse, tfidf_vectorizer, tfidf_vectorizer_attributes = reccomendationAllColumns(productos,userProducts, text_columns, numeric_columns, atributes_columns, n_components, keyWords, stemming)
        #Hay que aplicar un peso asociado segun la cantidad de columnas numericas y textuales
        # print (len(text_columns))
        # print (len(numeric_columns))
        # print (len(atributes_columns))
        #Imprimir palabras mas ocupadas y su cantidad de veces
        #print (tfidf_vectorizer.vocabulary_)
        #plotVectorizedData(tfidf_vectorizer, text_features_products, text_features_users)
        total_columns = len(text_columns) + len(numeric_columns) + len(atributes_columns)
        text_weight = (len(text_columns) + len(atributes_columns))/total_columns
        numeric_weight = len(numeric_columns)/total_columns
        attribute_weight = len(atributes_columns)/total_columns
        if (n_components != 0):
            attribute_weight = n_components/total_columns
        return  text_weight * text_similarity + numeric_weight * numeric_similarity + attribute_weight * attribute_similarity
    
def get_product(idProducto, productos):
    #Obtener la informacion del producto
    producto = productos[productos['idProducto'] == idProducto].iloc[0].values.tolist()
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

@app.get('/users/{idUsuario}')
async def read_user(idUsuario: str):
    #Se cargan los datos de los usuarios
    userProducts = pd.read_csv('data/usuarios.csv', sep=',')
    #Filtrar los productos de los usuarios ocupando la categoria de interes
    response = userProducts.to_json(orient="records")
    response = JSONResponse(content=response)
    return response

@app.get("/product/{idProducto}/{categoria}")
async def read_product(idProducto: str, categoria: str):
    #Se cargan los datos de los productos
    productos = pd.read_csv('data/categorie'+ categoria + '.csv', sep=',')
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
        print ("item", item)
        idProducto = item.idProducto
        categoria = item.categoria
        #Se cargan los datos de los usuarios
        userProducts = pd.read_csv('data/usuarios'+ categoria  + '.csv', sep=',')
        print (userProducts.shape)
        #Filtrar los productos de los usuarios ocupando la categoria de interes
        #Se cargan los datos de los productos
        productos = pd.read_csv('data/categorie'+ categoria + '.csv', sep=',')
        #Buscar el producto
        producto = productos[productos['idProducto'] == idProducto].iloc[0]
        #Agregar el producto al usuario
        df_nueva_fila = pd.DataFrame([producto], columns=productos.columns)
        userProducts = pd.concat([userProducts, df_nueva_fila], ignore_index=True)
        print ("nuevo data frame", userProducts.shape)
        print (userProducts)
        #Guardar los datos
        userProducts.to_csv('data/usuarios'+ categoria  + '.csv', index=False)
        
        return "success"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Change to post
class Item(BaseModel):
    categoria: str
    cantidad: int
    keyWords: list = None
    stemming: bool = False
    n_components: int = 0

@app.post("/recommendations")
async def read_root(item: Item):
    #Se cargan los datos de los usuarios
    #Usar model_dumps para transformar a json
    #Obtener categoria desde item
    try:
        print("datos de item", item)
        categoriaInteres = item.categoria
        cantidad = item.cantidad
        keyWords = item.keyWords
        stemming = item.stemming
        n_components = item.n_components
        #Usar los datos para cargar segun la categoria
        #Se cargan los datos de los usuarios
        userProducts = pd.read_csv('data/usuarios' +  categoriaInteres  + '.csv', sep=',')
        #Filtrar los productos de los usuarios ocupando la categoria de interes
        #Se cargan los datos de los productos
        productos = pd.read_csv('data/categorie'+ categoriaInteres +'.csv', sep=',')
        print ("categirua", categoriaInteres)
        print (userProducts.shape)
        print (productos.shape)
        with open('data/categoriesRegistersComplete.json', 'r') as archivo:
            data = json.load(archivo)
        datosCategoria = data[categoriaInteres]
        atributes_columns = []
        for atributo in datosCategoria['attributes']:
            atributes_columns.append(atributo['id'])

        cantidad = cantidad + 1
        similitud_combinada = get_reccomendation(productos,userProducts, text_columns, numeric_columns, atributes_columns,  n_components, keyWords,stemming  )
        print ("similitud combinada", similitud_combinada)
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
                "categoria": categoriaInteres
            })
        response = {
            "categoria": categoriaInteres,
            "cantidad": cantidad-1,
            "productos": listaRespuesta
        }
        response = JSONResponse(content=response)
        return response
 
        # for i in range(similitud_combinada.shape[0]):
        #     similar_items = [(productos['idProducto'].iloc[j], similitud_combinada[i][j]) for j in similar_indices]
        #     similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        #     #Transformar a JSON
        #     listaRespuesta = list()
        #     for item in similar_items:
        #         #Transform to json
        #         listaRespuesta.append({
        #             "idProducto": item[0],
        #             "similitud": item[1],
        #             "categoria": categoriaInteres
        #         })
        #     response = {
        #         "categoria": categoriaInteres,
        #         "cantidad": cantidad-1,
        #         "productos": listaRespuesta
        #     }
        #     response = JSONResponse(content=response)
        #     return response
    except Exception as e:
        print ("error", e)
        raise HTTPException(status_code=500, detail=str(e))
    
