# Clasificador de poses de mano

Este proyecto implementa un sistema de clasificacion de poses de mano con Deep Learning en Python y PyTorch. El modelo recibe imagenes reales de manos y las clasifica en una de cuatro categorias:

- `mano_abierta`
- `puno`
- `paz`
- `pulgar_arriba`

El proyecto compara dos enfoques de entrenamiento: una CNN propia entrenada desde cero y una ResNet18 con transferencia de aprendizaje. Ambos enfoques se entrenan con tres tamanos dinamicos de dataset para analizar como cambia el desempeno al usar mas o menos imagenes.

## Objetivo

El sistema busca construir y evaluar un clasificador multiclase de poses de mano. Para ello, el proyecto:

- Construye un dataset de imagenes reales organizado por carpetas.
- Implementa una CNN propia para clasificacion multiclase.
- Implementa una ResNet18 usando pesos preentrenados y fine-tuning.
- Aplica data augmentation y regularizacion.
- Evalua el desempeno mediante metricas de entrenamiento y validacion.
- Analiza diferencias entre modelos y errores de clasificacion.

## Dataset

Las imagenes se almacenan en `dataset/`, separadas por categoria:

```text
dataset/
|-- mano_abierta/
|-- puno/
|-- paz/
`-- pulgar_arriba/
```

El proyecto calcula automaticamente cuantos archivos existen en cada carpeta. Con el dataset actual, las clases tienen:

| Clase | Imagenes disponibles |
| --- | ---: |
| `mano_abierta` | 1133 |
| `paz` | 1129 |
| `pulgar_arriba` | 528 |
| `puno` | 314 |

Los tres tamanos de entrenamiento se calculan de forma dinamica:

| Tamano | Criterio |
| --- | --- |
| `pequeno` | Usa aproximadamente 1/3 de cada carpeta |
| `mediano` | Usa aproximadamente 2/3 de cada carpeta |
| `grande` | Usa todas las imagenes disponibles de cada carpeta |

Con los datos actuales, la division queda asi:

| Tamano | Total usado | Entrenamiento | Validacion | Prueba |
| --- | ---: | ---: | ---: | ---: |
| `pequeno` | 1033 | 721 | 153 | 159 |
| `mediano` | 2068 | 1446 | 308 | 314 |
| `grande` | 3104 | 2171 | 464 | 469 |

La division de validacion y prueba no se usa para ajustar pesos durante el entrenamiento. Esas imagenes se reservan para medir si el modelo aprende a generalizar y no solamente a memorizar las imagenes vistas.

## Estructura

```text
Entrenamiento_manos/
|-- dataset/
|-- imagen_prueba/
|-- models/
|-- results/
|-- src/
|   |-- train_cnn.py
|   |-- train_resnet.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- predict_folder.py
|   `-- utils.py
|-- requirements.txt
`-- README.md
```

## Instalacion

El proyecto se ejecuta en Windows con un entorno virtual de Python:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si el equipo tiene una GPU NVIDIA, PyTorch con CUDA puede acelerar el entrenamiento:

```powershell
.\venv\Scripts\pip.exe install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

La disponibilidad de GPU se verifica con:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Si PowerShell bloquea la activacion del entorno virtual, se habilita la ejecucion local con:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Entrenamiento CNN

La CNN propia se entrena desde cero. Esta arquitectura aprende directamente del dataset del proyecto, sin pesos preentrenados.

Dataset pequeno:

```powershell
.\venv\Scripts\python.exe src/train_cnn.py --dataset dataset --dataset_size pequeno --epochs 20 --batch_size 8
```

Dataset mediano:

```powershell
.\venv\Scripts\python.exe src/train_cnn.py --dataset dataset --dataset_size mediano --epochs 20 --batch_size 8
```

Dataset grande:

```powershell
.\venv\Scripts\python.exe src/train_cnn.py --dataset dataset --dataset_size grande --epochs 20 --batch_size 8
```

Archivos generados:

- `models/cnn_pequeno.pth`
- `models/cnn_mediano.pth`
- `models/cnn_grande.pth`
- `results/cnn_pequeno_training_metrics.csv`
- `results/cnn_mediano_training_metrics.csv`
- `results/cnn_grande_training_metrics.csv`
- `results/cnn_pequeno_training.png`
- `results/cnn_mediano_training.png`
- `results/cnn_grande_training.png`

## Entrenamiento ResNet18

ResNet18 se entrena mediante transferencia de aprendizaje. El modelo parte de pesos preentrenados en ImageNet, reemplaza la capa final para clasificar las cuatro poses del proyecto y luego aplica fine-tuning en `layer4`.

Dataset pequeno:

```powershell
.\venv\Scripts\python.exe src/train_resnet.py --dataset dataset --dataset_size pequeno --epochs 20 --batch_size 8
```

Dataset mediano:

```powershell
.\venv\Scripts\python.exe src/train_resnet.py --dataset dataset --dataset_size mediano --epochs 20 --batch_size 8
```

Dataset grande:

```powershell
.\venv\Scripts\python.exe src/train_resnet.py --dataset dataset --dataset_size grande --epochs 20 --batch_size 8
```

Archivos generados:

- `models/resnet_pequeno.pth`
- `models/resnet_mediano.pth`
- `models/resnet_grande.pth`
- `results/resnet_pequeno_training_metrics.csv`
- `results/resnet_mediano_training_metrics.csv`
- `results/resnet_grande_training_metrics.csv`
- `results/resnet_pequeno_training.png`
- `results/resnet_mediano_training.png`
- `results/resnet_grande_training.png`

## Evaluacion

La evaluacion calcula accuracy, precision, recall, f1-score y matriz de confusion usando el conjunto de prueba correspondiente al tamano indicado.

CNN pequena:

```powershell
.\venv\Scripts\python.exe src/evaluate.py --model models/cnn_pequeno.pth --architecture cnn --dataset dataset --dataset_size pequeno
```

CNN mediana:

```powershell
.\venv\Scripts\python.exe src/evaluate.py --model models/cnn_mediano.pth --architecture cnn --dataset dataset --dataset_size mediano
```

CNN grande:

```powershell
.\venv\Scripts\python.exe src/evaluate.py --model models/cnn_grande.pth --architecture cnn --dataset dataset --dataset_size grande
```

ResNet pequena:

```powershell
.\venv\Scripts\python.exe src/evaluate.py --model models/resnet_pequeno.pth --architecture resnet --dataset dataset --dataset_size pequeno
```

ResNet mediana:

```powershell
.\venv\Scripts\python.exe src/evaluate.py --model models/resnet_mediano.pth --architecture resnet --dataset dataset --dataset_size mediano
```

ResNet grande:

```powershell
.\venv\Scripts\python.exe src/evaluate.py --model models/resnet_grande.pth --architecture resnet --dataset dataset --dataset_size grande
```

## Prediccion

El script `predict.py` clasifica una imagen individual y genera una visualizacion con la imagen, la etiqueta predicha y los porcentajes de cada categoria:

```powershell
.\venv\Scripts\python.exe src/predict.py --model models/resnet_grande.pth --image imagen_prueba\ejemplo.jpg
```

El script `predict_folder.py` clasifica todas las imagenes de una carpeta. No genera CSV de predicciones; su salida principal es una carpeta con imagenes anotadas:

```powershell
.\venv\Scripts\python.exe src/predict_folder.py --model models/resnet_grande.pth --folder imagen_prueba
```

Salida por defecto:

```text
results/resnet_grande_imagen_prueba_visualizations/
```

Cada imagen anotada muestra:

- La imagen original.
- La etiqueta con mayor porcentaje.
- El porcentaje de confianza de la categoria ganadora.
- Los porcentajes de las demas categorias.

El umbral para marcar una prediccion como `no_seguro` es 60% por defecto. Se puede cambiar asi:

```powershell
.\venv\Scripts\python.exe src/predict_folder.py --model models/resnet_grande.pth --folder imagen_prueba --confidence_threshold 0.70
```

Si se requiere ocultar las predicciones individuales en consola:

```powershell
.\venv\Scripts\python.exe src/predict_folder.py --model models/resnet_grande.pth --folder imagen_prueba --hide_predictions
```

## Resultados

Los resultados actuales se obtuvieron desde los archivos `*_training_metrics.csv` almacenados en `results/`. El ranking se ordena por el mejor `val_accuracy` alcanzado durante el entrenamiento.

| Puesto | Modelo | Mejor epoca | Mejor `val_accuracy` | `val_loss` |
| ---: | --- | ---: | ---: | ---: |
| 1 | `resnet_mediano` | 10 | 97.40% | 0.3270 |
| 2 | `resnet_grande` | 20 | 96.98% | 0.3098 |
| 3 | `resnet_pequeno` | 17 | 95.42% | 0.3544 |
| 4 | `cnn_grande` | 19 | 90.73% | 0.2967 |
| 5 | `cnn_pequeno` | 19 | 90.20% | 0.4146 |
| 6 | `cnn_mediano` | 20 | 86.69% | 0.4264 |

Segun estos resultados, `resnet_mediano` obtiene el mejor valor puntual de validacion. Sin embargo, `resnet_grande` tambien presenta un rendimiento muy alto y termina el entrenamiento con el mejor resultado de su ultima epoca.

## Analisis de resultados

ResNet18 supera a la CNN propia porque utiliza transferencia de aprendizaje. Al iniciar con pesos preentrenados, ResNet ya contiene filtros utiles para reconocer bordes, texturas, formas y patrones visuales generales. Posteriormente, el fine-tuning adapta esas representaciones a las poses de manos del proyecto.

La CNN propia, en cambio, aprende desde cero. Esto la hace mas dependiente de la cantidad y variedad del dataset, y por eso obtiene resultados mas bajos. Aunque la CNN logra clasificar las poses, tiene menor capacidad para distinguir detalles finos entre categorias visualmente parecidas, como `puno`, `pulgar_arriba` y `mano_abierta`.

En terminos generales, el proyecto muestra que:

- ResNet18 es el enfoque mas efectivo para este dataset.
- El uso de fine-tuning mejora la adaptacion del modelo a poses especificas.
- El balanceo de clases ayuda a compensar categorias con menos imagenes, especialmente `puno`.
- Las CNN entrenadas desde cero pueden funcionar, pero requieren mas datos y mas ajuste para acercarse al desempeno de ResNet.

## Detalles tecnicos

- Entrada del modelo en `224x224`.
- Division del dataset: 70% entrenamiento, 15% validacion y 15% prueba.
- Data augmentation aplicado solamente en entrenamiento.
- Muestreo balanceado en CNN y ResNet para compensar clases con menos imagenes.
- Funcion de perdida: `CrossEntropyLoss`.
- Regularizacion en ResNet mediante `label_smoothing`.
- Optimizador CNN: `Adam`.
- Optimizador ResNet: `AdamW`.
- ResNet guarda el mejor checkpoint segun validacion.
- El dispositivo se selecciona automaticamente con `torch.cuda.is_available()`.
