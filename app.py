import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, session, flash
from flask_socketio import SocketIO, emit
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
from ia_modelo import predecir_prioridad

load_dotenv()

app = Flask(__name__)
# contraseña para mantener sesiones seguras
app.secret_key = 'llave_super_secreta_de_taskserv' 

@app.before_request
def proteger_rutas():
    # 1. Definir qué rutas SÍ pueden ver los usuarios que no han iniciado sesión
    # Agrega aquí tu ruta de login y los archivos estáticos (CSS/JS)
    rutas_publicas = ['login', 'static'] 
    
    # request.endpoint contiene el nombre de la función que el usuario quiere visitar
    if 'usuario_id' not in session and request.endpoint not in rutas_publicas:
        # 2. Si no está logueado y la ruta no es pública, lo mandamos al login
        return redirect(url_for('login'))
        
@app.route('/')
def index():
    return render_template('index.html')
    
# Inicializamos el  (WebSockets)
socketio = SocketIO(app)

DATABASE_URL = os.environ["DATABASE_URL"]

def obtener_conexion():
    return psycopg2.connect(DATABASE_URL)

def inicializar_bd():
    try:
        
        conexion = obtener_conexion() 
        
        with conexion:
            with conexion.cursor() as cursor:
                
                # Crear tabla usuarios
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS usuarios (
                        id SERIAL PRIMARY KEY,
                        nombre_usuario VARCHAR(50) UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    );
                ''')
                
                # Crear tabla tareas
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tareas (
                        id SERIAL PRIMARY KEY,
                        descripcion TEXT NOT NULL,
                        estado VARCHAR(20) DEFAULT 'Pendiente',
                        usuario_id INTEGER NOT NULL,
                        FOREIGN KEY (usuario_id) REFERENCES usuarios (id) ON DELETE CASCADE
                    );
                ''')
                cursor.execute("ALTER TABLE tareas ADD COLUMN IF NOT EXISTS prioridad TEXT DEFAULT 'Media';")

        # Guardar cambios y cerrar conexion con la BD
        conexion.commit()
        cursor.close()
        conexion.close()

    except Exception as e:
        
        print(f"Error detectado al inicializar la base de datos: {e}")

inicializar_bd()

# --- RUTAS DE ACCESO (LOGIN Y REGISTRO) ---
@app.route('/')
def inicio():
    if 'usuario_id' in session:
        return redirect('/tareas')
    return render_template('login.html')

@app.route('/registro', methods=['POST'])
def registro():
    nombre = request.form.get('nombre_usuario')
    password = request.form.get('password')

    if not nombre or not password:
        return redirect('/')

    password_hash = generate_password_hash(password)

    conexion = obtener_conexion()
    cursor = conexion.cursor()

    try:
        cursor.execute(
            'INSERT INTO usuarios (nombre_usuario, password) VALUES (%s, %s)',
            (nombre, password_hash)
        )
        conexion.commit()

        flash('¡Registro exitoso! Ahora puedes iniciar sesión.', 'success')

    except psycopg2.IntegrityError:
        conexion.rollback()
        flash('Ese nombre de usuario ya existe.', 'error')

    finally:
        cursor.close()
        conexion.close()

    return redirect('/')


@app.route('/login', methods=['POST'])
def login():
    nombre = request.form.get('nombre_usuario').strip()
    password = request.form.get('password', '')
    
    conexion = obtener_conexion()
    cursor = conexion.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('SELECT * FROM usuarios WHERE nombre_usuario = %s', (nombre,))
    usuario = cursor.fetchone()
    
    cursor.close()
    conexion.close()
    
    if usuario and check_password_hash(usuario['password'], password):
        session['usuario_id'] = usuario['id']
        session['nombre'] = usuario['nombre_usuario']
        return redirect('/tareas')
    else:
        flash('Usuario o contraseña incorrectos.', 'error')
        return redirect('/')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# --- RUTAS DE TAREAS (PROTEGIDAS Y EN TIEMPO REAL) ---
@app.route('/tareas')
def tareas():
    if 'usuario_id' not in session:
        return redirect('/')
        
    conexion = obtener_conexion()
    cursor = conexion.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('SELECT * FROM tareas WHERE usuario_id = %s ORDER BY id ASC', (session['usuario_id'],))
    tareas_db = cursor.fetchall()
    
    cursor.close()
    conexion.close()
    
    return render_template('tareas.html', tareas_html=tareas_db)

@app.route('/agregar_tarea', methods=['POST'])
def agregar_tarea():
    if 'usuario_id' not in session:
        return redirect('/')
        
    nueva_desc = request.form.get('descripcion')
    mi_id = session['usuario_id'] 
    
    prioridad_ia = predecir_prioridad(nueva_desc)
    
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    
    cursor.execute('INSERT INTO tareas (descripcion, usuario_id, prioridad) VALUES (%s, %s, %s)', (nueva_desc, mi_id, prioridad_ia))
    conexion.commit()
    
    cursor.close()
    conexion.close()
    
    # Aqui nos avisa que hay una tarea nueva 
    socketio.emit('actualizacion_tareas', {'mensaje': '¡Alguien agregó una tarea!'})
    
    return redirect('/tareas')

@app.route('/eliminar_tarea/<int:id>')
def eliminar_tarea(id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    
    cursor.execute('DELETE FROM tareas WHERE id = %s', (id,))
    conexion.commit()
    
    cursor.close()
    conexion.close()
    
    # Aqui nos avisa que se elmino una tarea!
    socketio.emit('actualizacion_tareas', {'mensaje': '¡Alguien eliminó una tarea!'})
    
    return redirect('/tareas')

@app.route('/completar_tarea/<int:id>')
def completar_tarea(id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    
    cursor.execute("UPDATE tareas SET estado = 'Completada' WHERE id = %s", (id,))
    conexion.commit()
    
    cursor.close()
    conexion.close()
    
    
    socketio.emit('actualizacion_tareas', {'mensaje': '¡Alguien completó una tarea!'})
    
    return redirect('/tareas')

if __name__ == '__main__':
    
    socketio.run(app, debug=True)
