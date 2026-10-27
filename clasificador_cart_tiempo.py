import re
from functools import lru_cache

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

# ------------------------------------------------------------- Texto -------
stemmer = SnowballStemmer("spanish")
STOPWORDS = {"de", "del", "la", "las", "el", "los", "un", "una", "en", "o", "y",
             "a", "al", "por", "para", "con", "se", "que", "es", "esta", "su", "hay"}
_QUITAR_ACENTOS = str.maketrans("áéíóúü", "aeiouu")  # conserva la ñ


def normalizar(texto):
    return texto.lower().translate(_QUITAR_ACENTOS)


@lru_cache(maxsize=None)
def _raiz(palabra):
    return stemmer.stem(palabra)


# ---- Detección de tiempo ----
_PATRON_NUM = re.compile(r"(\d+(?:[.,]\d+)?)\s*(minutos?|mins?|horas?|hrs?|hs?|dias?|semanas?|meses|mes)\b")
_FRASES = [
    (re.compile(r"\burgente\b|\burge\b|\binmediat[oa]\b|\bde inmediato\b|\bahora mismo\b"), "t_inmediato"),
    (re.compile(r"\bhoy\b|\bmañana\b|\bmanana\b"), "t_hoy"),
    (re.compile(r"\besta semana\b"), "t_semana"),
    (re.compile(r"sin fecha|sin prisa|sin urgencia|cuando haya tiempo|tiempo de sobra|proximo mes"), "t_largo"),
]


def _a_minutos(numero, unidad):
    n = float(numero.replace(",", "."))
    if unidad.startswith("min"):
        return n
    if unidad.startswith("h"):
        return n * 60
    if unidad.startswith("d"):
        return n * 1440
    if unidad.startswith("sem"):
        return n * 10080
    return n * 43200  # mes / meses


def _categoria(minutos):
    if minutos <= 240:
        return "t_inmediato"
    if minutos <= 1440:
        return "t_hoy"
    if minutos <= 10080:
        return "t_semana"
    return "t_largo"


def detectar_tiempo(texto):
    """Devuelve las categorías de tiempo encontradas, p. ej. ['t_inmediato']."""
    t = normalizar(texto)
    cats = [_categoria(_a_minutos(n, u)) for n, u in _PATRON_NUM.findall(t)]
    cats += [cat for patron, cat in _FRASES if patron.search(t)]
    return list(dict.fromkeys(cats))  # sin repetidos, conserva el orden


def analizador_de_raices(texto):
    """Raíces de las palabras (sin stopwords ni números) + categorías de tiempo."""
    palabras = re.findall(r"[a-zñ0-9]+", normalizar(texto))
    raices = [_raiz(p) for p in palabras if p not in STOPWORDS and not p.isdigit()]
    return raices + detectar_tiempo(texto)


# ------------------------------------------------------------- Datos -------
# OJO: las etiquetas de las frases con tiempo (las últimas 42) las asigné yo siguiendo
# la lógica "plazo corto + falla grave = Alta". Reemplázalas por tu criterio real.
ALTA = [
    "Fallo crítico en el servidor de producción sistema caído",
    "La base de datos de usuarios no responde",
    "Hackeo detectado o brecha de seguridad",
    "Error critico en la base de datos",
    "Se cayó el sistema de pagos",
    "Servidor apagado o sin conexion",
    "Levantar servicios caidos de emergencia",
    "Correo corporativo caído para toda la empresa, urgente",
    "Ataque de ransomware en los equipos, actuar en 15 minutos",
    "Sitio web de ventas inaccesible, el tiempo de caída cuesta dinero, urgente",
    "Falla eléctrica en el centro de datos, resolver en 1 hora",
    "Servidor de respaldo corrupto, requiere atención inmediata",
    "Pérdida de datos de clientes, atender de inmediato",
    "Firewall comprometido, solución en menos de 2 horas",
    "Red principal sin servicio, restablecer en 45 minutos",
    "Base de datos de facturación bloqueada, tiempo crítico, urgente",
    "Acceso no autorizado al servidor, contener en 1 hora",
    "Sistema de nómina caído, hay que resolverlo en 3 horas",
    "Aplicación móvil no abre para ningún usuario, urgente",
    "Servidor DNS caído, restaurar en 1 hora",
    "Falla de energía en el servidor principal, restablecer en 20 minutos",
]
# (7 originales sin tiempo + 14 nuevas con tiempo)
MEDIA = [
    "Configurar los nuevos switches de red",
    "Instalar actualizaciones de Windows",
    "Revisar conexion wifi intermitente",
    "Programar backend del nuevo modulo",
    "Crear nueva tabla en postgres",
    "Respaldar informacion de la empresa",
    "Levantar entorno de pruebas local",
    "Actualizar antivirus en las computadoras, en 2 días",
    "Migrar usuarios al nuevo dominio, plazo de 1 semana",
    "Corregir error en el reporte mensual, entregar hoy",
    "Instalar impresora de red en el área de contabilidad, en 3 días",
    "Optimizar consultas lentas de la base de datos, tiempo estimado 4 días",
    "Restablecer contraseñas de un grupo de usuarios, hoy",
    "Configurar VPN para trabajo remoto, esta semana",
    "Revisar logs del servidor por errores recurrentes, en 2 días",
    "Cambiar disco duro de una laptop, plazo de 24 horas",
    "Probar el nuevo módulo de reportes antes de la entrega, en 5 días",
    "Instalar certificado SSL en el sitio, tiempo disponible 3 días",
    "Revisar lentitud del sistema en el área de ventas, mañana",
    "Actualizar inventario de equipos de cómputo, esta semana",
    "Actualizar firmware del router, en 2 días",
]
BAJA = [
    "Comprar cartuchos de toner para impresora",
    "Diseñar un nuevo boton gris para el menu",
    "Limpiar el polvo de teclados y pantallas",
    "Cambiar cable ethernet dañado",
    "Acomodar los escritorios del area",
    "Pintar la oficina de soporte",
    "Cambiar el fondo de pantalla de las computadoras, sin prisa",
    "Reemplazar un mouse viejo, en 3 semanas",
    "Etiquetar los cables del rack, cuando haya tiempo",
    "Ordenar el almacén de repuestos, sin fecha límite",
    "Rotular las computadoras con el nombre del usuario, en 1 mes",
    "Cambiar el color del logotipo en la intranet, sin urgencia",
    "Comprar mousepads para el equipo, el próximo mes",
    "Reubicar la impresora a otra mesa, sin prisa",
    "Actualizar el directorio telefónico interno, en 2 semanas",
    "Cambiar la decoración de la sala de juntas, sin fecha límite",
    "Limpiar el cuarto de herramientas, cuando haya tiempo",
    "Reemplazar una silla dañada, en 3 semanas",
    "Archivar facturas antiguas de papel, tiempo de sobra",
    "Revisar el inventario de cables sobrantes, en 1 mes",
]

X_train = ALTA + MEDIA + BAJA
y_train = ["Alta"] * len(ALTA) + ["Media"] * len(MEDIA) + ["Baja"] * len(BAJA)


# ------------------------------------------------------------- Modelo ------
def construir_busqueda():
    pipeline = Pipeline([
        ("vectorizador", CountVectorizer(analyzer=analizador_de_raices, binary=True)),
        # CART: árbol binario con criterio gini (implementación de scikit-learn)
        ("cart", DecisionTreeClassifier(criterion="gini", class_weight="balanced", random_state=42)),
    ])
    rejilla = {
        "cart__max_depth": [None, 3, 4, 5, 8],
        "cart__min_samples_leaf": [1, 2, 3],
        "cart__ccp_alpha": [0.0, 0.005, 0.01, 0.02],  # poda por costo-complejidad (propia de CART)
    }
    cv_interno = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    return GridSearchCV(pipeline, rejilla, scoring="f1_macro", cv=cv_interno, n_jobs=-1)


def evaluar():
    etiquetas = ["Alta", "Media", "Baja"]
    cv_externo = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(construir_busqueda(), X_train, y_train, cv=cv_externo)
    print("=== Evaluación honesta (validación cruzada anidada) ===")
    print(classification_report(y_train, pred, labels=etiquetas, zero_division=0))
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(confusion_matrix(y_train, pred, labels=etiquetas),
                       index=etiquetas, columns=etiquetas), "\n")


def entrenar():
    busqueda = construir_busqueda().fit(X_train, y_train)
    print("Mejores hiperparámetros:", {k.split("__")[1]: v for k, v in busqueda.best_params_.items()})
    return busqueda.best_estimator_


def mostrar_arbol(modelo):
    vec, cart = modelo.named_steps["vectorizador"], modelo.named_steps["cart"]
    nombres = vec.get_feature_names_out()
    print("\n=== Reglas aprendidas por el árbol ===")
    print(export_text(cart, feature_names=list(nombres)))
    imp = pd.Series(cart.feature_importances_, index=nombres).sort_values(ascending=False)
    print("Palabras más importantes (las que empiezan con t_ son de TIEMPO):")
    print(imp[imp > 0].head(10).round(3).to_string(), "\n")


def predecir(modelo, descripcion, umbral=0.6):
    vec, cart = modelo.named_steps["vectorizador"], modelo.named_steps["cart"]
    X = vec.transform([descripcion])
    tiempo = detectar_tiempo(descripcion)
    if X.nnz == 0:  # ninguna raíz conocida: no adivinamos
        return {"prioridad": "Revisar manualmente", "confianza": 0.0, "tiempo_detectado": tiempo}
    proba = cart.predict_proba(X)[0]
    i = int(proba.argmax())
    return {
        "prioridad": str(cart.classes_[i]) if proba[i] >= umbral else "Revisar manualmente",
        "confianza": round(float(proba[i]), 2),
        "tiempo_detectado": tiempo,
    }


if __name__ == "__main__":
    print(f"Ejemplos de entrenamiento: {len(X_train)} "
          f"({len(ALTA)} Alta, {len(MEDIA)} Media, {len(BAJA)} Baja)\n")
    evaluar()
    modelo = entrenar()
    mostrar_arbol(modelo)

    pruebas = [
        "Servidor de correo caído, resolver en 90 minutos",
        "Cambiar un mouse que no funciona, sin prisa",
        "Instalar Office en la laptop nueva, en 3 días",
        "Revisar impresora que no imprime, hoy",
        "El sistema de pagos no responde",   # sin tiempo: decide por las palabras
        "xyz qwerty",
    ]
    for frase in pruebas:
        print(f"{frase!r} -> {predecir(modelo, frase)}")
