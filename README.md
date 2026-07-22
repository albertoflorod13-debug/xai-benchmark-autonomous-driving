# Benchmark de técnicas de explicabilidad para redes neuronales en conducción autónoma

TFM — Máster en Big Data, Inteligencia Artificial e Ingeniería de Datos.
**Autor:** Alberto Floro Rodríguez · **Tutores:** Javier del Ser Lorente, José Manuel García Nieto

## Descripción

Benchmark de métodos de explicabilidad (XAI) — D-CRISP y SSGrad-CAM++ — aplicados sobre
YOLO26 + KITTI, con cuantificación de incertidumbre vía Test-Time Augmentation (TTA) y
evaluación de fidelidad, localización y complejidad (Quantus + métricas propias de
detección). La evaluación de robustez y de estabilidad entre muestras está planificada pero
**pendiente de implementar** (ver "Estado actual").

## Estructura del repositorio

```
configs/
├── tta_uq.yaml                 # configuración de TTA / cuantificación de incertidumbre
└── xai/
    ├── ssgradcampp.yaml        # config de SSGrad-CAM++ (generación + evaluación)
    └── dcrisp.yaml             # config de D-CRISP (pendiente de uso)

data/kitti/
├── kitti.yaml                   # rutas y nombres de clase del dataset (no depende de la maquina)
└── kitti_local.example.yaml    # plantilla de kitti_local.yaml (ruta absoluta local, gitignored)

models/
├── pretrained/                 # pesos preentrenados en COCO (yolo26n.pt, gitignored)
└── finetuned/                  # pesos tras el fine-tuning sobre KITTI (best.pt)

notebooks/
├── tests.ipynb                       # pruebas exploratorias rapidas, al margen del pipeline final:
│                                      # tiempo real por epoca del fine-tuning, efecto del resize/rect, etc.
├── tta_augmentation_selection.ipynb  # protocolo de seleccion de las augmentations usadas en TTA
├── tta_uncertainty.ipynb             # test de TTA/UQ sobre 30 imagenes de validacion
├── tta_uq_analysis.ipynb             # analisis de incertidumbre TTA/UQ sobre las 1496 imagenes de validacion
├── ss_gradcampp_test.ipynb           # validacion de SSGrad-CAM++ adaptado a YOLO26
└── xai_evaluation_analysis.ipynb     # analisis de las metricas de evaluacion (fidelidad/localizacion/complejidad)

results/
├── runs/                       # salidas brutas de Ultralytics (gitignored)
├── ssgradcampp/                 # heatmaps + eval_metrics.csv de SSGrad-CAM++ (gitignored)
├── tta_uq/                     # resultados de incertidumbre (gitignored)


scripts/
├── train_finetune_stage1.py    # fine-tuning de YOLO26 sobre KITTI, etapa 1 (backbone congelado)
├── train_finetune_stage2.py    # fine-tuning de YOLO26 sobre KITTI, etapa 2 (todo descongelado)
├── run_tta_uq.py                # cuantificacion de incertidumbre (TTA) sobre el conjunto de validacion
├── run_xai_explanations.py     # generacion de heatmaps SSGrad-CAM++ sobre el conjunto de validacion
└── run_evaluation.py            # evaluacion de los heatmaps generados (fidelidad/localizacion/complejidad)

src/xai_benchmark/              
├── data/                       # carga y conversión de etiquetas KITTI
├── detection/                  # utilidades sobre la cabeza Detect de YOLO26
├── uncertainty/                # TTA / cuantificación de incertidumbre
├── xai/                        # métodos de explicabilidad (SSGrad-CAM++, D-CRISP)
└── evaluation/                 # métricas de evaluación de explicaciones
```

## Estado actual

| Objetivos | Estado |
|---|---|
| 1. Transfer learning de YOLO26 sobre KITTI | Cerrado (fine-tuning en dos etapas ejecutado) |
| 2. Aplicación de métodos XAI | SSGrad-CAM++ implementado y ejecutado sobre todo el conjunto de validación. D-CRISP **pendiente** |
| 3. Evaluación de las explicaciones | Fidelidad (Deletion/Insertion), localización (Pointing Game/EBPG) y complejidad (Sparseness) implementadas y ejecutadas sobre SSGrad-CAM++. Robustez y estabilidad entre muestras **pendientes**.|
| 4. Conclusiones | Pendiente (depende de cerrar D-CRISP) |

Además, ya cerrado aunque no es uno de los 4 objetivos formales: cuantificación de
incertidumbre vía TTA (`src/xai_benchmark/uncertainty/tta.py`), ejecutada sobre las 1496
imágenes de validación.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate    
pip install -r requirements.txt
```


