# Machine Learning III - Práctica 2: Bayesian Logistic Regression

## Instalación

1. **Clonar el repositorio**.

2. **Crear un entorno virtual**:
   ```bash
   python -m venv p2_venv
   .\p2_venv\Scripts\activate
   ```

3. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### 1. Entrenamiento

Para entrenar el modelo, ejecuta el script `aprendizaje.py`. Este script:
1. Carga y procesa los datos si no existen.
2. Entrena el modelo Bayesiano MCMC.
3. Guarda la traza del modelo en `model/model_trace.npy`.

```bash
python aprendizaje.py
```

### 2. Inferencia y Evaluación

Una vez entrenado el modelo, puedes evaluar su rendimiento en el conjunto de prueba ejecutando `inferencia.py`. Este script cargará la traza guardada y calculará la precisión (accuracy) del modelo.

```bash
python inferencia.py
```
