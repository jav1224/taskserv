import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from clasificador_cart_tiempo import X_train as X_MANUAL, analizador_de_raices, detectar_tiempo, normalizar
from clasificador_cart_tiempo import y_train as Y_MANUAL
from generar_datos import generar

RUTA_REAL = Path("incidencias_reales.csv")
RUTA_SINTETICO = Path("incidencias_sinteticas.csv")
RUTA_MODELO = Path("modelo_optimo.joblib")
ETIQUETAS = ["Alta", "Media", "Baja"]

CONFIGURACIONES = [
    {"nombre": "CART solo reales",              "modelo": "cart",    "sint": False, "peso_real": 1,  "hoja": 1},
    {"nombre": "Bagging solo reales",           "modelo": "bagging", "sint": False, "peso_real": 1,  "hoja": 1},
    {"nombre": "CART reales+sintéticos (x5)",   "modelo": "cart",    "sint": True,  "peso_real": 5,  "hoja": 2},
    {"nombre": "Bagging reales+sint (x1)",      "modelo": "bagging", "sint": True,  "peso_real": 1,  "hoja": 2},
    {"nombre": "Bagging reales+sint (x5)",      "modelo": "bagging", "sint": True,  "peso_real": 5,  "hoja": 2},
    {"nombre": "Bagging reales+sint (x20)",     "modelo": "bagging", "sint": True,  "peso_real": 20, "hoja": 2},
]


# ------------------------------------------------------------- Datos -------
def cargar_reales():
    if not RUTA_REAL.exists():
        pd.DataFrame({"descripcion": X_MANUAL, "prioridad": Y_MANUAL}).to_csv(RUTA_REAL, index=False)
        print(f"[!] No existía {RUTA_REAL}: creé una plantilla con las {len(X_MANUAL)} frases escritas a mano")
        print("    (20 tuyas + 42 que escribí yo). Reemplázala/amplíala con incidencias REALES.\n")
    df = pd.read_csv(RUTA_REAL).dropna(subset=["descripcion", "prioridad"])
    df["prioridad"] = df["prioridad"].str.strip().str.capitalize()
    malas = set(df["prioridad"]) - set(ETIQUETAS)
    if malas:
        raise ValueError(f"Prioridades no válidas en {RUTA_REAL}: {malas}. Usa Alta, Media o Baja.")
    conteo = df["prioridad"].value_counts()
    if len(conteo) < 3 or conteo.min() < 5:
        raise ValueError("Necesito al menos 5 incidencias reales por cada prioridad para evaluar bien.")
    return df["descripcion"].astype(str).tolist(), df["prioridad"].tolist()


def cargar_sinteticos():
    if not RUTA_SINTETICO.exists():
        generar().to_csv(RUTA_SINTETICO, index=False)
    df = pd.read_csv(RUTA_SINTETICO).dropna()
    return df["descripcion"].astype(str).tolist(), df["prioridad"].tolist()


def sin_fuga(textos_sint, etiquetas_sint, textos_prueba):
    """Quita del entrenamiento sintético todo lo que contenga la frase base de una frase de prueba."""
    bases = {normalizar(t).split(",")[0].strip() for t in textos_prueba}
    claves = [normalizar(t) for t in textos_sint]
    keep = [i for i, c in enumerate(claves) if not any(b in c for b in bases)]
    return [textos_sint[i] for i in keep], [etiquetas_sint[i] for i in keep]


# ------------------------------------------------------------- Modelo ------
def construir_modelo(cfg, n_arboles):
    base = DecisionTreeClassifier(criterion="gini", class_weight="balanced",
                                  min_samples_leaf=cfg["hoja"], random_state=42)
    if cfg["modelo"] == "cart":
        return base
    return BaggingClassifier(base, n_estimators=n_arboles, bootstrap=True, n_jobs=-1, random_state=42)


def ajustar(cfg, t_real, y_real, t_sint, y_sint, n_arboles=100):
    textos, etiquetas, pesos = list(t_real), list(y_real), [cfg["peso_real"]] * len(t_real)
    if cfg["sint"]:
        textos += t_sint
        etiquetas += y_sint
        pesos += [1] * len(t_sint)
    vec = CountVectorizer(analyzer=analizador_de_raices, binary=True)
    X = vec.fit_transform(textos)
    modelo = construir_modelo(cfg, n_arboles).fit(X, etiquetas, sample_weight=np.array(pesos, dtype=float))
    return vec, modelo


def evaluar_configuraciones(t_real, y_real, t_sint, y_sint):
    y_arr = np.array(y_real)
    k = int(min(5, pd.Series(y_real).value_counts().min()))
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=3, random_state=42)
    filas, predicciones = [], {}
    for cfg in CONFIGURACIONES:
        f1s, accs, y_true_all, y_pred_all = [], [], [], []
        for idx_tr, idx_te in cv.split(t_real, y_arr):
            tr, te = [t_real[i] for i in idx_tr], [t_real[i] for i in idx_te]
            ytr, yte = list(y_arr[idx_tr]), list(y_arr[idx_te])
            ts, ys = sin_fuga(t_sint, y_sint, te) if cfg["sint"] else ([], [])
            vec, modelo = ajustar(cfg, tr, ytr, ts, ys)
            pred = modelo.predict(vec.transform(te))
            f1s.append(f1_score(yte, pred, average="macro", labels=ETIQUETAS, zero_division=0))
            accs.append(accuracy_score(yte, pred))
            y_true_all += yte
            y_pred_all += list(pred)
        filas.append({"configuración": cfg["nombre"], "f1 macro": np.mean(f1s), "± desv.": np.std(f1s),
                      "exactitud": np.mean(accs)})
        predicciones[cfg["nombre"]] = (y_true_all, y_pred_all)
    tabla = pd.DataFrame(filas).sort_values(["f1 macro", "± desv."], ascending=[False, True]).reset_index(drop=True)
    return tabla, predicciones


# ------------------------------------------------------------- Uso ---------
def importancias(vec, modelo, n=12):
    if hasattr(modelo, "estimators_"):
        imp = np.mean([t.feature_importances_ for t in modelo.estimators_], axis=0)
    else:
        imp = modelo.feature_importances_
    s = pd.Series(imp, index=vec.get_feature_names_out()).sort_values(ascending=False)
    return s[s > 0].head(n).round(3)


def predecir(vec, modelo, descripcion, umbral=0.55):
    X = vec.transform([descripcion])
    tiempo = detectar_tiempo(descripcion)
    if X.nnz == 0:
        return {"prioridad": "Revisar manualmente", "confianza": 0.0, "tiempo_detectado": tiempo}
    proba = modelo.predict_proba(X)[0]
    i = int(proba.argmax())
    return {"prioridad": str(modelo.classes_[i]) if proba[i] >= umbral else "Revisar manualmente",
            "confianza": round(float(proba[i]), 2), "tiempo_detectado": tiempo}


def main():
    t_real, y_real = cargar_reales()
    t_sint, y_sint = cargar_sinteticos()
    print(f"Incidencias reales: {len(t_real)} | sintéticas: {len(t_sint)}\n")

    tabla, preds = evaluar_configuraciones(t_real, y_real, t_sint, y_sint)
    print("=== Comparación (validación cruzada repetida; la prueba usa solo incidencias reales) ===")
    print(tabla.round(3).to_string(index=False), "\n")

    mejor = next(c for c in CONFIGURACIONES if c["nombre"] == tabla.loc[0, "configuración"])
    print(f"Configuración elegida: {mejor['nombre']}\n")
    y_true, y_pred = preds[mejor["nombre"]]
    print(classification_report(y_true, y_pred, labels=ETIQUETAS, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=ETIQUETAS),
                       index=ETIQUETAS, columns=ETIQUETAS), "\n")

    # Modelo final: todo lo disponible, más árboles si es bagging
    vec, modelo = ajustar(mejor, t_real, y_real, t_sint, y_sint, n_arboles=300)
    joblib.dump({"vectorizador": vec, "modelo": modelo, "configuracion": mejor}, RUTA_MODELO)
    print(f"Modelo final guardado en {RUTA_MODELO}")
    print("\nTérminos más importantes (t_ = tiempo):")
    print(importancias(vec, modelo).to_string(), "\n")

    for frase in ["Servidor de correo caído, resolver en 90 minutos",
                  "Cambiar un mouse que no funciona, sin prisa",
                  "Falla crítica en la base de datos, plazo de 1 semana",
                  "Revisar impresora que no imprime, hoy",
                  "xyz qwerty"]:
        print(f"{frase!r} -> {predecir(vec, modelo, frase)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # modo predicción con el modelo guardado
        guardado = joblib.load(RUTA_MODELO)
        print(predecir(guardado["vectorizador"], guardado["modelo"], " ".join(sys.argv[1:])))
    else:
        main()
