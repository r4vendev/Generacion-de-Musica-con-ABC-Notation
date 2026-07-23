# Generación de Música en Notación ABC con LSTM

Proyecto final de la maestría — generación de música en notación ABC carácter por carácter usando una red neuronal recurrente **LSTM**, con conversión final a audio real (WAV).

**Autor:** Kevin Cárdenas

📄 Informe completo (paper): [`paper/paper.pdf`](./paper/paper.pdf)

## Descripción

El proyecto entrena un modelo LSTM que aprende a generar secuencias de texto en notación ABC (un formato estándar para representar partituras musicales como texto). A partir de una semilla inicial, el modelo genera nuevas piezas musicales carácter por carácter, que luego se validan sintácticamente y se convierten en audio.

**Pipeline:**
1. Descarga y exploración del dataset (tunes en notación ABC, colección de O'Neill's — música irlandesa)
2. Preprocesamiento: vocabulario de caracteres + ventana de contexto con overlapping
3. Definición del modelo LSTM
4. Entrenamiento principal: 20 épocas fijas (sin early stopping)
5. Gráfica de épocas vs. loss
6. Generación de una canción nueva (sampling con temperatura)
7. Validación sintáctica del ABC generado (`music21`)
8. Conversión a audio: ABC → MIDI → WAV (`abc2midi` + `fluidsynth`)
9. Extra: segunda corrida del mismo modelo con early stopping, para comparar

## Arquitectura del modelo

- Capa de *embedding*: 128 dimensiones
- 2 capas LSTM apiladas, 256 unidades ocultas
- Dropout: 0.2
- Capa lineal de salida (proyección al tamaño del vocabulario)
- Función de pérdida: `CrossEntropyLoss`
- Optimizador: Adam (`lr=3e-3`)

## Dataset

[`abc-notation-of-tunes`](https://www.kaggle.com/datasets/raj5287/abc-notation-of-tunes) (Kaggle) — tunes de música irlandesa en notación ABC. Se descarga automáticamente al correr el notebook mediante `kagglehub` (no requiere configurar credenciales manualmente).

## Requisitos

Librerías de Python:
```
torch
numpy
matplotlib
music21
kagglehub
```

Herramientas del sistema (para la conversión a audio):
```
abcmidi
fluidsynth
fluid-soundfont-gm
```

Instalación rápida (incluida también en la primera celda del notebook):
```bash
pip install music21 kagglehub torch numpy matplotlib
apt-get install -y abcmidi fluidsynth fluid-soundfont-gm
```

## Estructura del repositorio

```
.
├── abc_music_generation_Kevin_Cárdenas.ipynb   # Notebook principal (pipeline completo)
├── paper/
│   └── paper.pdf                                # Informe en LaTeX (formato artículo científico)
├── checkpoints/
│   ├── LSTM_last.pt                             # Último checkpoint del entrenamiento principal
│   ├── LSTM_best.pt                             # Mejor checkpoint (menor val_loss) - entrenamiento principal
│   ├── LSTM_ES_last.pt                          # Último checkpoint (con early stopping)
│   └── LSTM_ES_best.pt                          # Mejor checkpoint (con early stopping)
├── audio/
│   ├── generated_song.wav                       # Audio generado - entrenamiento principal (20 épocas)
│   └── generated_song_es.wav                    # Audio generado - entrenamiento con early stopping
├── plots/
│   ├── epocas_vs_loss_lstm.png                  # Gráfica épocas vs. loss - entrenamiento principal
│   └── epocas_vs_loss_early_stopping.png        # Gráfica épocas vs. loss - con early stopping
└── README.md
```

## Cómo correrlo

1. Abrir el notebook `abc_music_generation_Kevin_Cárdenas.ipynb`.
2. Correr las celdas en orden — el notebook detecta automáticamente si existe GPU disponible.
3. El notebook fue ejecutado originalmente en un servidor H200 de la universidad; también corre en Colab (detecta la ruta base automáticamente y usa la carpeta actual si no encuentra la ruta del servidor).
4. Si ya existen checkpoints guardados en `checkpoints/`, el entrenamiento se reanuda automáticamente desde la última época completada en vez de empezar desde cero.

## Resultados

- Entrenamiento principal: 20 épocas fijas, sin early stopping.
- Entrenamiento extra: mismo modelo con early stopping (`patience=5`, techo máximo de 50 épocas), incluido para comparar en qué época se detiene por su cuenta.
- La generación usa *sampling* con temperatura (`temperature=0.8`) en vez de elegir siempre el carácter más probable, para que cada pieza generada sea distinta.
- Validación sintáctica de las piezas generadas mediante `music21`.

Ver el informe completo en [`paper/paper.pdf`](./paper/paper.pdf) para el análisis detallado, limitaciones y trabajo futuro.
