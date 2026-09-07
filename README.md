# Benchmark de técnicas de explicabilidad para redes neuronales en conducción autónoma

TFM — Máster en Big Data, Inteligencia Artificial e Ingeniería de Datos.
**Autor:** Alberto Floro Rodríguez · **Tutores:** Javier del Ser Lorente, José Manuel García Nieto

## Descripción

Benchmark comparativo de técnicas de explicabilidad (XAI) para redes neuronales de detección
de objetos en conducción autónoma: D-CRISP y SSGrad-CAM++, frente a una línea base de ruido
aleatorio, aplicados sobre YOLO26 fine-tuned en KITTI. Incluye cuantificación de incertidumbre
vía Test-Time Augmentation (TTA), y una evaluación exhaustiva de fidelidad (Deletion/Insertion
AUC, Minimal Subset), localización (Pointing Game, Energy-Based Pointing Game, Relevance Rank
Accuracy), complejidad (Sparseness), robustez a perturbaciones y estabilidad entre muestras.

El proyecto se extiende además con un caso de uso industrial de logística de almacén (dataset
LOCO): un robot planifica su ruta combinando un LLM y RRT-Connect, replanifica en tiempo real
ante conflictos detectados por visión, y genera un informe explicando cada conflicto con las
mismas técnicas XAI — ver `industrial_use_case/README.md` para el detalle completo.

## Estructura del repositorio

```
configs/
├── tta_uq.yaml                        # configuración de TTA / cuantificación de incertidumbre
└── xai/
    ├── ssgradcampp.yaml               # config de SSGrad-CAM++ (generación + evaluación)
    ├── dcrisp.yaml                    # config de D-CRISP (generación + evaluación)
    └── random_baseline.yaml           # config de la línea base de ruido aleatorio

data/kitti/
├── kitti.yaml                         # rutas y nombres de clase del dataset 
└── kitti_local.example.yaml           # plantilla de kitti_local.yaml 

models/
├── pretrained/                        # pesos preentrenados en COCO (yolo26n.pt, gitignored)
├── finetuned/                         # pesos tras el fine-tuning sobre KITTI (best.pt)
└── finetuned_loco/                    # pesos tras el fine-tuning sobre LOCO (gitignored, ver industrial_use_case/README.md)

notebooks/
├── tests.ipynb                        # pruebas exploratorias rápidas, al margen del pipeline final
├── tta_augmentation_selection.ipynb   # protocolo de selección de las augmentations usadas en TTA
├── tta_uncertainty.ipynb              # test de TTA/UQ sobre 30 imágenes de validación
├── tta_uq_analysis.ipynb              # análisis de incertidumbre TTA/UQ sobre las 1496 imágenes de validación
├── ss_gradcampp_test.ipynb            # validación de SSGrad-CAM++ adaptado a YOLO26
├── dcrisp_test.ipynb                  # validación de D-CRISP adaptado a YOLO26
├── metrics_test.ipynb                 # validación de las métricas de evaluación
├── xai_evaluation_analysis.ipynb      # primer análisis de SSGrad-CAM++ cruzado con TTA/UQ
└── final_analysis.ipynb               # análisis final consolidado: D-CRISP vs SSGrad-CAM++ vs baseline

scripts/
├── train_finetune_stage1.py           # fine-tuning de YOLO26 sobre KITTI, etapa 1 (backbone congelado)
├── train_finetune_stage2.py           # fine-tuning de YOLO26 sobre KITTI, etapa 2 (todo descongelado)
├── train_finetune_stage1_loco.py      # mismo esquema de fine-tuning, sobre LOCO (caso de uso industrial)
├── train_finetune_stage2_loco.py
├── convert_loco_to_yolo.py            # convierte las anotaciones COCO de LOCO a formato YOLO
├── run_tta_uq.py                      # cuantificación de incertidumbre (TTA) sobre el conjunto de validación
├── run_xai_explanations.py            # generación de heatmaps SSGrad-CAM++ sobre el conjunto de validación
├── run_dcrisp_explanations.py         # generación de heatmaps D-CRISP sobre el conjunto de validación
├── run_random_baseline_heatmaps.py    # generación de la línea base de ruido aleatorio
├── run_evaluation.py                  # evaluación de los heatmaps generados (fidelidad/localización/complejidad/robustez/estabilidad)
└── classify_distance_size.py          # clasificación de objetos por tamaño y lejanía real (análisis estratificado)

src/xai_benchmark/
├── data/                              # carga y conversión de etiquetas KITTI
├── detection/                         # utilidades sobre la cabeza Detect de YOLO26
├── uncertainty/                       # TTA / cuantificación de incertidumbre
├── xai/                               # métodos de explicabilidad (SSGrad-CAM++, D-CRISP)
└── evaluation/                        # métricas de evaluación de explicaciones

industrial_use_case/                   # extensión: planificación de rutas + XAI para un caso de uso de
                                        # logística de almacén -- ver su propio README.md para el detalle

results/                               # salidas de todos los scripts anteriores (gitignored, reproducible en su totalidad)
```

## Resultados

Comparación visual de las dos técnicas XAI del benchmark -- D-CRISP y SSGrad-CAM++ -- explicando
el mismo objeto: el coche más cercano a la cámara (16.39 m) en la imagen `000605` del conjunto
de validación de KITTI. En verde, la caja detectada por YOLO26; en rojo/amarillo, mayor
relevancia del heatmap para esa predicción.

<p align="center">
  <img src="results/final/readme_heatmaps/ssgradcampp_000605_nearest_car.png" width="75%">
  <br>
  <img src="results/final/readme_heatmaps/dcrisp_000605_nearest_car.png" width="75%">
</p>

Parámetros usados, idénticos a las ejecuciones finales sobre las 1496 imágenes de validación
(`configs/xai/ssgradcampp.yaml` y `configs/xai/dcrisp.yaml`):

| SSGrad-CAM++ | valor |
|---|---|
| checkpoint | `models/finetuned/best.pt` |
| conf_thres / iou_thres_nms | 0.25 / 0.5 |
| iou_match_thres | 0.999 |
| margin (M^{k,det}) | 0 (una única celda de la rejilla) |
| eps | 1e-8 |

| D-CRISP | valor |
|---|---|
| checkpoint | `models/finetuned/best.pt` |
| conf_thres / iou_thres_nms | 0.25 / 0.5 |
| n_masks (N) | 1000 |
| alpha | 0.50 |
| resolution | 16 |
| p1 | 0.25 |
| num_levels | 5 |


