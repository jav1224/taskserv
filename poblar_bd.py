import requests
import time


URL_LOGIN = "http://127.0.0.1:5000/login"
URL_AGREGAR = "http://127.0.0.1:5000/agregar_tarea"


MI_USUARIO = "Hugo" 
MI_PASSWORD = "1234"

# Los 35 elementos de prueba 
elementos = [
    "Servidor principal caído y sin respuesta",
    "Fallo crítico en la pasarela de pagos",
    "Error al iniciar sesión en el portal de clientes",
    "Base de datos de producción no responde",
    "Lentitud extrema en la carga de la página principal",
    "Actualización de software fallida en servidores",
    "Pérdida de conexión con la API externa de facturación",
    "Error 500 en la sección de reportes",
    "Interfaz de usuario desalineada en móviles",
    "Correos de confirmación no se están enviando",
    "Fuga de memoria en el servicio en segundo plano",
    "Disco duro del servidor al 99% de capacidad",
    "Error de sincronización de datos entre sucursales",
    "Certificado SSL expirado en el dominio principal",
    "Usuarios reportan cobros duplicados",
    "Problema de configuración en el firewall",
    "Caída inesperada del servidor de pruebas",
    "Error al exportar reportes a formato PDF",
    "Botón de compra no funciona en navegador Safari",
    "Posible vulnerabilidad de inyección SQL",
    "Fallo en el balanceador de carga",
    "Reinicio inesperado del servidor de aplicaciones",
    "Tiempo de respuesta de API supera los 10 segundos",
    "Error de permisos al subir archivos adjuntos",
    "La caché de la aplicación no se limpia",
    "Credenciales de base de datos comprometidas",
    "Fallo masivo en notificaciones push",
    "Datos no se guardan al editar el perfil",
    "Error 404 recurrente en la documentación",
    "Bloqueo masivo de cuentas por falso positivo",
    "Problema de enrutamiento en la red interna",
    "Error de validación en formulario de registro",
    "El sistema de respaldo automático falló",
    "Integración con proveedor de envíos falla",
    "Carga de imágenes en el catálogo muy lenta"
]

print("Iniciando sesión en Taskserv...")
# Creamos una sesión web para mantener el login abierto
s = requests.Session()
s.post(URL_LOGIN, data={'nombre_usuario': MI_USUARIO, 'password': MI_PASSWORD})

print("Iniciando la inserción de 35 elementos...\n")

for i, tarea in enumerate(elementos, 1):
    
    s.post(URL_AGREGAR, data={'descripcion': tarea})
    print(f"[{i}/35] Éxito: {tarea}")
    time.sleep(0.3)

print("\n¡Listo! Revisa tu panel de control.")