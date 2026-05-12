# Clasificador de poses de mano

Proyecto de Deep Learning en Python con PyTorch para clasificar imagenes reales de manos en cuatro clases:

- `mano_abierta`
- `puno`
- `paz`
- `pulgar_arriba`

El entrenamiento y la evaluacion usan exclusivamente las imagenes propias ubicadas en `dataset/`. ResNet18 puede iniciar con pesos preentrenados de ImageNet, pero su ultima capa se reemplaza para clasificar solamente las 4 clases del proyecto.

## Entrenamientos

El proyecto permite ejecutar cuatro entrenamientos:

1. CNN propia con dataset pequeno: hasta 25 imagenes por clase.
2. CNN propia con dataset mediano: hasta 50 imagenes por clase.
3. ResNet18 con dataset pequeno: hasta 25 imagenes por clase.
4. ResNet18 con dataset mediano: hasta 50 imagenes por clase.

Si una clase aun no tiene suficientes imagenes, el codigo usa las imagenes disponibles y muestra un aviso en consola. Esto permite avanzar mientras se completa el dataset.

## Estructura

```text
clasificador_manos/
|-- dataset/
|   |-- mano_abierta/
|   |-- puno/
|   |-- paz/
|   `-- pulgar_arriba/
|-- src/
|   |-- train_cnn.py
|   |-- train_resnet.py
|   |-- evaluate.py
|   `-- utils.py
|-- models/
|-- results/
|-- requirements.txt
`-- README.md
```

## Instalacion en Windows con venv

Desde la carpeta del proyecto:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea la activacion del entorno virtual, ejecuta:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Entrenar CNN propia

Dataset pequeno:

```powershell
python src/train_cnn.py --dataset dataset --dataset_size small --epochs 20 --batch_size 8
```

Dataset mediano:

```powershell
python src/train_cnn.py --dataset dataset --dataset_size medium --epochs 20 --batch_size 8
```

Modelos generados:

- `models/cnn_small.pth`
- `models/cnn_medium.pth`

Graficas generadas:

- `results/cnn_small_training.png`
- `results/cnn_medium_training.png`

## Entrenar ResNet18

Dataset pequeno:

```powershell
python src/train_resnet.py --dataset dataset --dataset_size small --epochs 20 --batch_size 8
```

Dataset mediano:

```powershell
python src/train_resnet.py --dataset dataset --dataset_size medium --epochs 20 --batch_size 8
```

Modelos generados:

- `models/resnet_small.pth`
- `models/resnet_medium.pth`

Graficas generadas:

- `results/resnet_small_training.png`
- `results/resnet_medium_training.png`

## Evaluar modelos

CNN pequena:

```powershell
python src/evaluate.py --model models/cnn_small.pth --architecture cnn --dataset dataset --dataset_size small
```

CNN mediana:

```powershell
python src/evaluate.py --model models/cnn_medium.pth --architecture cnn --dataset dataset --dataset_size medium
```

ResNet pequena:

```powershell
python src/evaluate.py --model models/resnet_small.pth --architecture resnet --dataset dataset --dataset_size small
```

ResNet mediana:

```powershell
python src/evaluate.py --model models/resnet_medium.pth --architecture resnet --dataset dataset --dataset_size medium
```

La evaluacion calcula accuracy, precision, recall, f1-score y matriz de confusion. Los resultados se guardan en `results/`.

## Detalles tecnicos

- Imagenes redimensionadas a `224x224`.
- Funcion de perdida: `CrossEntropyLoss`.
- Optimizador: `Adam`.
- Division del dataset: 70% entrenamiento, 15% validacion y 15% prueba.
- Data augmentation solamente en entrenamiento.
- Deteccion automatica de GPU con `torch.cuda.is_available()`.
