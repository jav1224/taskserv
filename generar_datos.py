import random

import pandas as pd

SEMILLA = 42
N_EJEMPLOS = 8000
RUIDO = 0.05  # 5 % de etiquetas "humanamente discutibles"

# ----------------------------------------------------------- Problemas -----
SISTEMAS_CRITICOS = [
    "el servidor de producción", "la base de datos de clientes", "el sistema de pagos",
    "el correo corporativo", "la red principal", "el firewall", "el servidor DNS",
    "el sitio web de ventas", "la aplicación móvil", "el sistema de facturación",
    "el servidor de respaldo", "el centro de datos", "el sistema de nómina",
    "el servidor de autenticación",
]
PLANTILLAS_CRITICAS = [
    "Se cayó {s}", "{S} no responde", "Falla crítica en {s}", "Brecha de seguridad en {s}",
    "Ransomware detectado en {s}", "Pérdida de datos en {s}", "{S} caído para todos los usuarios",
    "Acceso no autorizado a {s}", "Error crítico en {s}", "{S} apagado o sin conexión",
]
TAREAS_MEDIAS = [
    "Configurar los nuevos switches de red", "Actualizar el antivirus de las computadoras",
    "Configurar la VPN para trabajo remoto", "Instalar el certificado SSL del sitio",
    "Actualizar el firmware del router", "Crear una tabla nueva en postgres",
    "Probar el módulo de reportes", "Levantar el entorno de pruebas local",
    "Restablecer contraseñas de un grupo de usuarios", "Cambiar el disco duro de una laptop",
    "Migrar usuarios al nuevo dominio", "Revisar conexión wifi intermitente",
    "Revisar logs del servidor por errores recurrentes", "Optimizar consultas lentas de la base de datos",
    "Programar el backend del nuevo módulo", "Respaldar información de la empresa",
    "Instalar actualizaciones de Windows", "Corregir un error en el reporte mensual",
]
TAREAS_MENORES = [
    "Comprar cartuchos de toner para impresora", "Limpiar el polvo de teclados y pantallas",
    "Cambiar el cable ethernet dañado", "Acomodar los escritorios del área",
    "Pintar la oficina", "Etiquetar los cables del rack", "Ordenar el almacén de repuestos",
    "Reubicar la impresora a otra mesa", "Cambiar el fondo de pantalla de las computadoras",
    "Reemplazar una silla dañada", "Comprar mousepads para el equipo",
    "Diseñar un botón gris nuevo para el menú", "Archivar facturas antiguas de papel",
    "Cambiar el color del logotipo en la intranet", "Actualizar el directorio telefónico interno",
    "Cambiar la decoración de la sala de juntas", "Reemplazar un mouse viejo",
]
AREAS = ["ventas", "contabilidad", "soporte", "recursos humanos", "almacén", "dirección"]


# -------------------------------------------------------------- Tiempo -----
def _u(n, singular, plural):
    return f"{n} {singular if n == 1 else plural}"


def frase_tiempo(cat, rng):
    if cat == "inmediato":
        n = rng.randint(1, 4)
        return rng.choice([
            "urgente", "de inmediato", "ahora mismo", "atender de inmediato",
            "tiempo crítico, urgente",
            f"en {rng.randint(10, 120)} minutos",
            f"en {_u(n, 'hora', 'horas')}",
            f"en menos de {_u(n, 'hora', 'horas')}",
            f"plazo de {_u(n, 'hora', 'horas')}",
        ])
    if cat == "hoy":
        return rng.choice(["hoy", "hoy mismo", "para mañana", f"en {rng.randint(5, 24)} horas",
                           "plazo de 24 horas"])
    if cat == "semana":
        n = rng.randint(2, 7)
        return rng.choice(["esta semana", f"en {n} días", "plazo de 1 semana",
                           f"tiempo estimado {n} días", f"plazo de {n} días"])
    if cat == "largo":
        n = rng.randint(2, 4)
        return rng.choice(["sin prisa", "sin fecha límite", "cuando haya tiempo", "el próximo mes",
                           f"en {n} semanas", "en 1 mes", "tiempo de sobra", "sin urgencia"])
    return ""


# ------------------------------------------------------ Regla de prioridad --
TABLA_PRIORIDAD = {
    # (gravedad, tiempo): prioridad
    (2, "inmediato"): "Alta", (2, "hoy"): "Alta", (2, None): "Alta",
    (2, "semana"): "Media", (2, "largo"): "Media",
    (1, "inmediato"): "Alta", (1, "hoy"): "Media", (1, "semana"): "Media",
    (1, None): "Media", (1, "largo"): "Baja",
    (0, "inmediato"): "Media", (0, "hoy"): "Baja", (0, "semana"): "Baja",
    (0, None): "Baja", (0, "largo"): "Baja",
}
VECINA = {"Alta": ["Media"], "Baja": ["Media"], "Media": ["Alta", "Baja"]}
_QUITAR_ACENTOS = str.maketrans("áéíóú", "aeiou")


def generar(n=N_EJEMPLOS, semilla=SEMILLA, ruido=RUIDO):
    rng = random.Random(semilla)
    filas = []
    for _ in range(n):
        gravedad = rng.choices([2, 1, 0], weights=[0.35, 0.35, 0.30])[0]
        if gravedad == 2:
            s = rng.choice(SISTEMAS_CRITICOS)
            base = rng.choice(PLANTILLAS_CRITICAS).format(s=s, S=s[0].upper() + s[1:])
        elif gravedad == 1:
            base = rng.choice(TAREAS_MEDIAS)
        else:
            base = rng.choice(TAREAS_MENORES)

        cat = rng.choices([None, "inmediato", "hoy", "semana", "largo"],
                          weights=[0.25, 0.20, 0.20, 0.20, 0.15])[0]
        texto = base + (", " + frase_tiempo(cat, rng) if cat else "")

        texto = rng.choice(["", "", "", "Por favor, ", "Reporte: ", "Ticket abierto: "]) + texto
        if rng.random() < 0.25:
            texto += rng.choice([f" (área de {rng.choice(AREAS)})", f", reportado por {rng.choice(AREAS)}"])
        if rng.random() < 0.25:
            texto = texto.translate(_QUITAR_ACENTOS)  # gente que escribe sin acentos
        if rng.random() < 0.15:
            texto = texto.lower()

        etiqueta = TABLA_PRIORIDAD[(gravedad, cat)]
        if rng.random() < ruido:
            etiqueta = rng.choice(VECINA[etiqueta])
        filas.append((texto, etiqueta))
    return pd.DataFrame(filas, columns=["descripcion", "prioridad"])


if __name__ == "__main__":
    df = generar()
    df.to_csv("incidencias_sinteticas.csv", index=False)
    print(f"Generadas {len(df)} incidencias ({df['descripcion'].nunique()} frases distintas)")
    print(df["prioridad"].value_counts().to_string())
    print(df.sample(8, random_state=1).to_string(index=False))
