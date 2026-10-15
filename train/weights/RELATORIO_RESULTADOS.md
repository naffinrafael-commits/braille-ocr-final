# Resultados

Métricas ponta a ponta: uma célula conta como acerto quando a caixa coincide com a anotada (IoU de pelo menos 0,5) e o padrão de pontos está correto. CER: erros de caracteres na sequência lida, em relação à anotada (0 é perfeito).

### YOLO11n, 640 px: métricas do YOLO

| Domínio | Precisão | Revocação | Hmean | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| roboflow | 0.8655 | 0.9318 | 0.8974 | 0.9383 | 0.685 |
| angelina | 0.8449 | 0.7596 | 0.8 | 0.7895 | 0.1454 |
| sintetico | 0.9952 | 0.9943 | 0.9948 | 0.9948 | 0.9319 |
| mosaico | 0.9928 | 0.9735 | 0.9831 | 0.9928 | 0.9548 |

### YOLO11n, 640 px: leitura ponta a ponta

| Domínio | Precisão | Revocação | Hmean | Detecção sem classe | CER | FPS GPU |
|---|---|---|---|---|---|---|
| roboflow | 0.7228 | 0.9761 | 0.8306 | 0.9786 | 0.4077 | 58.54 |
| angelina | 0.7904 | 0.8037 | 0.797 | 0.8198 | 0.1129 | 38.51 |
| sintetico | 0.9983 | 0.9994 | 0.9989 | 0.9999 | 0.0166 | 24.11 |
| mosaico | 0.9646 | 0.9818 | 0.9731 | 0.9951 | 0.0371 | 42.5 |

### YOLO26n, 640 px: métricas do YOLO

| Domínio | Precisão | Revocação | Hmean | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| roboflow | 0.847 | 0.9101 | 0.8774 | 0.9206 | 0.6582 |
| angelina | 0.7884 | 0.7339 | 0.7602 | 0.7517 | 0.1368 |
| sintetico | 0.9947 | 0.9894 | 0.9921 | 0.9932 | 0.9064 |
| mosaico | 0.9909 | 0.9564 | 0.9733 | 0.9881 | 0.9407 |

### YOLO26n, 640 px: leitura ponta a ponta

| Domínio | Precisão | Revocação | Hmean | Detecção sem classe | CER | FPS GPU |
|---|---|---|---|---|---|---|
| roboflow | 0.6723 | 0.9724 | 0.7949 | 0.9815 | 0.5025 | 56.26 |
| angelina | 0.8307 | 0.8314 | 0.8311 | 0.8503 | 0.1315 | 38.6 |
| sintetico | 0.9939 | 0.9993 | 0.9966 | 0.9997 | 0.0268 | 27.93 |
| mosaico | 0.9599 | 0.9709 | 0.9654 | 0.9898 | 0.0476 | 44.59 |

### YOLO11n, 416 px: métricas do YOLO

| Domínio | Precisão | Revocação | Hmean | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| roboflow | 0.8436 | 0.8506 | 0.8471 | 0.9001 | 0.6371 |
| angelina | 0.7698 | 0.6371 | 0.6972 | 0.6823 | 0.1147 |
| sintetico | 0.982 | 0.9719 | 0.9769 | 0.9894 | 0.852 |
| mosaico | 0.993 | 0.945 | 0.9684 | 0.9845 | 0.8958 |

### YOLO11n, 416 px: leitura ponta a ponta

| Domínio | Precisão | Revocação | Hmean | Detecção sem classe | CER | FPS GPU |
|---|---|---|---|---|---|---|
| roboflow | 0.6799 | 0.9481 | 0.7919 | 0.9547 | 0.4765 | 58.18 |
| angelina | 0.83 | 0.8179 | 0.8239 | 0.8482 | 0.1547 | 42.16 |
| sintetico | 0.9872 | 0.9945 | 0.9908 | 0.9972 | 0.052 | 27.23 |
| mosaico | 0.9491 | 0.9594 | 0.9542 | 0.9814 | 0.0662 | 48.87 |

### 2 etapas (detector de células + MobileNetV2): leitura ponta a ponta

| Domínio | Precisão | Revocação | Hmean | Detecção sem classe | CER | FPS GPU |
|---|---|---|---|---|---|---|
| roboflow | 0.6913 | 0.9518 | 0.8009 | 0.9654 | 0.465 | 25.29 |
| angelina | 0.8545 | 0.8646 | 0.8595 | 0.8828 | 0.1189 | 14.78 |
| sintetico | 0.9996 | 0.9997 | 0.9996 | 1.0 | 0.0162 | 13.73 |
| mosaico | 0.9654 | 0.966 | 0.9657 | 1.0 | 0.0364 | 24.75 |

### Leitura por grade (método clássico): leitura ponta a ponta

| Domínio | Precisão | Revocação | Hmean | Detecção sem classe | CER | FPS GPU |
|---|---|---|---|---|---|---|
| roboflow | 0.0155 | 0.0161 | 0.0158 | 0.019 | 1.6674 | 1.97 |
| angelina | 0.0 | 0.0 | 0.0 | 0.0 | 0.9959 | 0.97 |
| sintetico | 0.9468 | 0.9589 | 0.9528 | 0.9874 | 0.0952 | 3.32 |
| mosaico | 0.4365 | 0.1659 | 0.2404 | 0.1659 | 0.8453 | 2.2 |

### Velocidade em CPU (imagens por segundo)

- YOLO11n, 640 px: 8.79
- YOLO26n, 640 px: 8.91
- YOLO11n, 416 px: 17.8
- 2 etapas (detector de células + MobileNetV2): 5.58
- Leitura por grade (método clássico): 1.8