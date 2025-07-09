
# CS:GO Tactical Analyzer - Predicción de Supervivencia

## Descripción

Este proyecto es una aplicación web interactiva que predice la probabilidad de supervivencia de un jugador en una ronda de CS:GO utilizando Machine Learning (Random Forest). Incluye una interfaz visual inspirada en la estética de CS:GO y una consola tipo terminal que simula el análisis táctico de la partida.

- **Backend:** Flask (Python)
- **Frontend:** HTML + CSS + JS (Jinja2)
- **Modelo ML:** RandomForestRegressor (scikit-learn)
- **Tema:** Edición Morada / Cyberpunk

---

## 🚀 Instalación y Ejecución

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/canunz/randomforest.git
   cd randomforest/csgo-predictor
   ```

2. **Crea y activa un entorno virtual (opcional pero recomendado):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # En Windows
   # source venv/bin/activate  # En Linux/Mac
   ```

3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Coloca tu archivo de datos CSV (opcional, para entrenar con tus datos):**
   - Copia tu archivo `datos_limpios_con_derivadas.csv` en la carpeta `csgo-predictor/`.

5. **Ejecuta la aplicación:**
   ```bash
   python app.py
   ```

6. **Abre tu navegador en:**  
   [http://localhost:5000](http://localhost:5000)


## 📦 Estructura del Proyecto

csgo-predictor/
├── app.py                  # Backend Flask
├── model.py                # Lógica del modelo ML
├── modelo/
│   └── modelo_entrenado.pkl # Modelo entrenado
├── templates/
│   ├── index.html          # Formulario
│   └── resultado.html      # Resultados
├── requirements.txt        # Dependencias
└── README.md               # Este archivo


## ✨ Créditos y Licencia

- Inspirado en CS:GO y la comunidad de Machine Learning.
- UI y diseño: Edición Morada / Cyberpunk.
- Licencia: MIT
