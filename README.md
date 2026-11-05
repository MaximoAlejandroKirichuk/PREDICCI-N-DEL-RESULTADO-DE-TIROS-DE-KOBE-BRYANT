# Predicción de tiros de Kobe Bryant

Red neuronal con TensorFlow y Keras para predecir si un tiro de básquetbol entró o no,
a partir de la posición en la cancha, el tipo de jugada y el contexto del partido.

Trabajo práctico de la materia Inteligencia Artificial.

## El problema

Clasificación binaria sobre 25.697 tiros de Kobe Bryant. Dadas las características de un
lanzamiento, predecir si fue convertido (`1`) o no (`0`).

El efecto de cada variable depende del valor de las demás. Por ejemplo, jugar en playoffs
no modifica el porcentaje de acierto global (44,64% contra 44,47%), pero sí lo modifica en
combinación con la zona y el momento del partido: bajo el aro cae de 62,24% a 59,09%, y en
los últimos dos minutos del cuarto período, de 39,02% a 35,24%.

Una regresión logística asigna un coeficiente fijo por variable y no puede representar esa
dependencia. La capa oculta de una red neuronal construye combinaciones no lineales de las
entradas y sí puede.

## Arquitectura

```
141 entradas  →  Dense(64, relu)  →  Dense(1, sigmoid)
```

9.153 parámetros. Una capa oculta, sin dropout. Esa configuración no se eligió por
criterio general: es la más simple de cuatro variantes medidas, y ninguna de las más
complejas mejoró el resultado.

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy | 0,6751 |
| Precision | 0,7089 |
| Recall | 0,4610 |
| F1-score | 0,5587 |
| ROC-AUC | 0,6905 |
| Log loss | 0,6141 |
| **Línea base** | **0,5539** |
| **Mejora sobre la línea base** | **+12,12 p.p.** |

La línea base es el *accuracy* de un clasificador que predijera siempre la clase
mayoritaria. Sin esa referencia, un 67,51% no es interpretable.

### Comparación de variantes

| Variante | Features | Params | Accuracy | ROC-AUC |
|---|---|---|---|---|
| Sin `action_type`, 64 | 86 | 5.633 | 0,6051 | 0,6176 |
| **Con `action_type`, 64** | **141** | **9.153** | **0,6751** | **0,6905** |
| Con `action_type`, 64 + dropout | 141 | 9.153 | 0,6745 | 0,6880 |
| Con `action_type`, 64-32 | 141 | 11.201 | 0,6745 | 0,6866 |

Conservar `action_type` aporta 7,00 puntos de *accuracy*. Agregar capacidad al modelo no
aporta nada: la variante con un 22% más de parámetros rinde igual.

El factor determinante del desempeño fue la información suministrada al modelo, no la
capacidad de la red.

## Límite del problema

El modelo no supera el 68% y la causa no está en la arquitectura. Los errores de mayor
confianza son volcadas desde el área restringida, con probabilidades entre 0,92 y 0,97,
que no entraron. Cuando una volcada falla es porque hubo un bloqueo o una falta, y no
existe ninguna variable en el conjunto que registre eso.

El modelo no puede predecir a partir de información que no recibe.

## Datos

*Kobe Bryant Shot Selection*, de Kaggle. El CSV no está en este repositorio.

Para ejecutar el notebook, descargar `data.csv` del dataset y dejarlo en la raíz del
proyecto, o subirlo cuando el notebook lo solicite al correr en Google Colab.

## Ejecución

El notebook está preparado para los dos entornos sin modificaciones: busca el CSV en el
directorio local y, si no lo encuentra, abre el selector de archivos de Colab.

**Google Colab:** subir `kobe_red_neuronal.ipynb`, conectar y ejecutar todas las celdas.

**Local:**

```bash
python3 -m venv .venv
.venv/bin/pip install tensorflow "pandas<3" scikit-learn matplotlib ipykernel
.venv/bin/python -m ipykernel install --user --name kobe-tf
```

Ejecutar el notebook con el kernel `kobe-tf`. La ejecución completa toma unos minutos: la
sección 12.4 entrena las cuatro variantes de la comparación.

## Reproducibilidad

El notebook fija la semilla 42 en NumPy, TensorFlow y Keras, y usa `random_state=42` en la
división de los datos. Su ejecución completa reproduce todos los valores de este
documento.
