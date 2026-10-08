from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk.stem.snowball import SnowballStemmer

# Creamos el extractor de raíces
stemmer = SnowballStemmer('spanish')

def analizador_de_raices(texto):
    """
    Esta función toma una frase, la limpia y reduce cada palabra a su raíz.
    Ejemplo: 'Limpieza de equipos' -> ['limpi', 'de', 'equip']
    """
    palabras = texto.lower().split()
    raices = [stemmer.stem(palabra) for palabra in palabras]
    return raices

#   Datos de entrenamiento 
X_train = [
    "Fallo crítico en el servidor de producción sistema caído",
    "La base de datos de usuarios no responde",
    "Hackeo detectado o brecha de seguridad",
    "Error critico en la base de datos",
    "Se cayó el sistema de pagos",
    "Servidor apagado o sin conexion",
    "Levantar servicios caidos de emergencia",

    "Configurar los nuevos switches de red",
    "Instalar actualizaciones de Windows",
    "Revisar conexion wifi intermitente",
    "Programar backend del nuevo modulo",
    "Crear nueva tabla en postgres",
    "Respaldar informacion de la empresa",
    "Levantar entorno de pruebas local",

    "Comprar cartuchos de toner para impresora",
    "Diseñar un nuevo boton gris para el menu",
    "Limpiar el polvo de teclados y pantallas",
    "Cambiar cable ethernet dañado",
    "Acomodar los escritorios del area",
    "Pintar la oficina de soporte"
]

y_train = [
    "Alta", "Alta", "Alta", "Alta", "Alta", "Alta", "Alta",
    "Media", "Media", "Media", "Media", "Media", "Media", "Media",
    "Baja", "Baja", "Baja", "Baja", "Baja", "Baja"
]

# 2. aqui se usa el analizador de raíces personalizado
vectorizador = CountVectorizer(analyzer=analizador_de_raices)
X_train_num = vectorizador.fit_transform(X_train)

# 3. (Algoritmo CART)
modelo_cart = DecisionTreeClassifier(criterion='gini', random_state=42)
modelo_cart.fit(X_train_num, y_train)

def predecir_prioridad(descripcion_tarea):
    """Recibe la tarea del usuario y predice la prioridad usando las raíces aprendidas"""
    texto_num = vectorizador.transform([descripcion_tarea])
    prediccion = modelo_cart.predict(texto_num)
    return prediccion[0]