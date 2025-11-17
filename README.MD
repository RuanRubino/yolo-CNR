# YOLOv11 CNR Park - Detecção de Vagas de Estacionamento

## 📋 Descrição

Este projeto implementa um sistema de **detecção de vagas de estacionamento ocupadas e disponíveis** utilizando **YOLOv11** com otimização de hiperparâmetros via **Optuna** (otimização bayesiana). O modelo foi treinado no dataset **CNRPark**, que contém 2.926 imagens anotadas em formato YOLO Oriented Object Detection.

## 🎯 Objetivo

Desenvolver um modelo de visão computacional capaz de:
- Detectar vagas de estacionamento em imagens
- Classificar vagas como **ocupadas (0)** ou **disponíveis (1)**
- Otimizar o desempenho através de busca de hiperparâmetros automatizada

## 📊 Dataset

### CNRPark v2
- **Total de imagens**: 2.926
- **Tamanho**: 640x640 pixels
- **Classes**: 2 (0: ocupado, 1: disponível)
- **Anotações**: YOLOv8 Oriented Object Detection Format

**Download do dataset**: https://universe.roboflow.com/projects-oqpad/cnrpark-hnvuk/dataset/2

### Pré-processamento aplicado:
- Auto-orientação de dados de pixel (remoção de EXIF-orientation)
- Redimensionamento para 640x640 (Stretch)

### Aumentação de dados:
- Probabilidade de flip horizontal: 50%
- Probabilidade de flip vertical: 50%
- Ajuste de brilho aleatório: -21% a +21%
- Desfoque Gaussiano aleatório: 0 a 1.7 pixels

## 🏗️ Estrutura do Projeto

```
yoloCNR/
├── data.yaml                 # Configuração do dataset
├── yolo11teste.py           # Script principal de treinamento com Optuna
├── README.MD                # Este arquivo
├── README.roboflow.txt      # Informações do Roboflow
├── train/                   # Dados de treinamento
│   ├── images/
│   └── labels/
├── valid/                   # Dados de validação
│   ├── images/
│   └── labels/
└── test/                    # Dados de teste
    ├── images/
    └── labels/
```

## 🚀 Como Usar

### 1. Pré-requisitos

```bash
pip install ultralytics optuna
```

### 2. Configurar o Dataset

Baixe o dataset do CNRPark:
1. Acesse: https://universe.roboflow.com/projects-oqpad/cnrpark-hnvuk/dataset/2
2. Extraia os arquivos na pasta do projeto
3. Verifique se o arquivo `data.yaml` está configurado corretamente

### 3. Executar o Treinamento com Otimização

```bash
# Treinamento com 30 trials de otimização
python yolo11teste.py --trials 30 --study-name meu_estudo

# Com armazenamento persistente (SQLite)
python yolo11teste.py --trials 30 --study-name meu_estudo --storage sqlite:///optuna.db

# Apenas treino final com melhores hiperparâmetros
python yolo11teste.py --final --study-name meu_estudo --storage sqlite:///optuna.db
```

### Argumentos disponíveis:
- `--trials`: Número de trials de otimização (padrão: 20)
- `--study-name`: Nome do estudo Optuna (padrão: yolov11_optuna)
- `--storage`: URI de armazenamento SQLite (ex: sqlite:///optuna.db)
- `--final`: Executar apenas o treino final com melhores hiperparâmetros

## ⚙️ Configurações de Treinamento

### Hiperparâmetros otimizados pelo Optuna:
- **Batch size**: [64, 128, 256]
- **Learning rate (lr0)**: 1e-10 a 1e-2
- **Optimizer**: SGD, Adam, Adamax
- **Weight decay**: 5e-5 a 5e-2

### Épocas:
- **Trials**: 5 épocas (avaliação rápida)
- **Treinamento final**: 300 épocas

### Modelo base:
- **YOLOv11 Nano** (yolo11n-obb.pt) - para detecção orientada

## 📈 Resultados

Os resultados do treinamento são salvos em `runs/optuna_yolov11/` com:
- Checkpoints do modelo
- Métricas de validação
- Arquivos de configuração

## 🔍 Estrutura do Script Principal

O script `yolo11teste.py` implementa:

1. **Função objetivo** (`objective`): 
   - Sugere hiperparâmetros via Optuna
   - Treina o modelo com poucos epochs
   - Valida e retorna mAP

2. **Execução Optuna** (`run_optuna`):
   - Cria estudo com maximização de mAP
   - Executa N trials
   - Salva melhores parâmetros

3. **Treinamento final** (`final_train`):
   - Treina modelo com melhores hiperparâmetros
   - 300 epochs para melhor desempenho

## 📝 Notas Importantes

- O script utiliza **GPU (device=0)** por padrão. Altere em `yolo11teste.py` se necessário
- Cada trial roda um treinamento curto para avaliação rápida de hiperparâmetros
- O treinamento final é automaticamente executado ao final
- Os resultados são persistentes se usar `--storage` com SQLite

## 🔗 Referências

- **YOLOv11**: https://github.com/ultralytics/ultralytics
- **Optuna**: https://optuna.org/
- **Dataset CNRPark**: https://universe.roboflow.com/projects-oqpad/cnrpark-hnvuk/dataset/2
- **Roboflow**: https://roboflow.com

## 📦 Dependências

```
ultralytics>=8.0.0
optuna>=3.0.0
```

## 🎓 Contexto

Projeto desenvolvido como parte de uma pesquisa em **Ciência da Computação** focando em:
- Detecção de objetos em tempo real
- Otimização de modelos de deep learning
- Aplicações de visão computacional em ambientes urbanos

---

**Dataset Source**: https://universe.roboflow.com/projects-oqpad/cnrpark-hnvuk/dataset/2