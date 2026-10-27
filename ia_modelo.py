from pathlib import Path

import joblib

RUTA_MODELO = Path(__file__).with_name("modelo_optimo.joblib")
PRIORIDADES_VALIDAS = {"Alta", "Media", "Baja"}
PRIORIDAD_POR_DEFECTO = "Media"  # coincide con DEFAULT 'Media' de la columna tareas.prioridad

_vectorizador = None
_modelo = None


def _entrenar_de_respaldo():
    """Entrena un modelo equivalente cuando no hay .joblib utilizable."""
    from clasificador_optimo import CONFIGURACIONES, ajustar, cargar_reales, cargar_sinteticos

    textos_reales, etiquetas_reales = cargar_reales()
    textos_sint, etiquetas_sint = cargar_sinteticos()
    config = next(c for c in CONFIGURACIONES if c["nombre"] == "Bagging reales+sint (x1)")
    return ajustar(config, textos_reales, etiquetas_reales, textos_sint, etiquetas_sint)


def _cargar_modelo():
    if RUTA_MODELO.exists():
        try:
            guardado = joblib.load(RUTA_MODELO)
            return guardado["vectorizador"], guardado["modelo"]
        except Exception as e:
            print(f"[ia_modelo] No pude cargar {RUTA_MODELO.name}: {e}. Entrenando de respaldo...")
    else:
        print(f"[ia_modelo] No existe {RUTA_MODELO.name}. Entrenando de respaldo...")
    return _entrenar_de_respaldo()


try:
    _vectorizador, _modelo = _cargar_modelo()
    print("[ia_modelo] Modelo de prioridad listo.")
except Exception as e:  # la app debe poder arrancar aunque el modelo falle
    print(f"[ia_modelo] ERROR al preparar el modelo: {e}. Se usará '{PRIORIDAD_POR_DEFECTO}'.")


def predecir_prioridad(descripcion_tarea):
    """Devuelve 'Alta', 'Media' o 'Baja'. Nunca lanza excepciones."""
    if _modelo is None or not descripcion_tarea or not str(descripcion_tarea).strip():
        return PRIORIDAD_POR_DEFECTO
    try:
        from clasificador_optimo import predecir

        resultado = predecir(_vectorizador, _modelo, str(descripcion_tarea))
        prioridad = resultado["prioridad"]
        # "Revisar manualmente" (modelo poco seguro) no existe en la BD ni en la plantilla
        return prioridad if prioridad in PRIORIDADES_VALIDAS else PRIORIDAD_POR_DEFECTO
    except Exception as e:
        print(f"[ia_modelo] Error al predecir: {e}")
        return PRIORIDAD_POR_DEFECTO
