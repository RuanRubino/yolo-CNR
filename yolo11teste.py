"""
Treinamento de YOLOv11 com Otimização Bayesiana (Optuna)

Como usar:
1) Ajuste os caminhos: DATA_YAML, WEIGHT_PRETRAIN (opcional), SAVE_DIR
2) Instale dependências: pip install ultralytics optuna
3) Rode: python treinamento_yolov11_optuna.py --trials 30 --study-name meu_estudo

Nota: este script usa a API `ultralytics.YOLO` (YOLOv11). Cada trial roda um treino curto para avaliar mAP; o melhor conjunto de hiperparâmetros é re-treinado com `final_epochs`.
"""

import argparse
import os
import json
import optuna
from ultralytics import YOLO

# --- Configurações ---
DATA_YAML = './data.yaml'  # seu arquivo .yaml do dataset
WEIGHT_PRETRAIN = 'yolo11n-obb.pt'  # checkpoint inicial
SAVE_DIR = 'runs/optuna_yolov11'

# parâmetros do estudo/treinamento final
TRIAL_EPOCHS = 5       # poucas épocas por trial para avaliar rapidamente
FINAL_EPOCHS = 300     # treinamento final com melhores hp

os.makedirs(SAVE_DIR, exist_ok=True)


def objective(trial: optuna.trial.Trial):
    """Objetivo do Optuna: sugere hiperparâmetros, treina e retorna mAP@0.5:0.95."""

    # Espaço de busca conforme solicitado
    batch = trial.suggest_categorical('batch', [64, 128, 256])
    lr0 = trial.suggest_loguniform('lr0', 1e-10, 1e-2)
    optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'Adamax'])
    weight_decay = trial.suggest_loguniform('weight_decay', 5e-5, 5e-2)
    epochs = 3

    run_name = f"optuna_trial_{trial.number}"
    run_dir = os.path.join(SAVE_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    model = YOLO(WEIGHT_PRETRAIN)

    train_args = dict(
        data=DATA_YAML,
        epochs=epochs,
        batch=batch,
        lr0=lr0,
        optimizer=optimizer,
        weight_decay=weight_decay,
        name=run_name,
        project=SAVE_DIR,
        exist_ok=True,
        device=0,
        imgsz = 640
    )

    print(f"Trial {trial.number}: batch={batch}, lr0={lr0:.1e}, optimizer={optimizer}, weight_decay={weight_decay:.1e}, epochs={epochs}")

    model.train(**train_args)

    val_results = model.val(data=DATA_YAML)

    mAP = None
    try:
        if hasattr(val_results, 'box'):
            mAP = float(val_results.box.map)
    except Exception:
        mAP = None

    if mAP is None and isinstance(val_results, dict):
        for k, v in val_results.items():
            if 'map' in k.lower():
                try:
                    mAP = float(v)
                    break
                except Exception:
                    pass

    if mAP is None:
        results_json = os.path.join(run_dir, 'results.json')
        if os.path.exists(results_json):
            try:
                with open(results_json, 'r') as f:
                    r = json.load(f)
                for k, v in r.items():
                    if 'map' in k.lower():
                        mAP = float(v)
                        break
            except Exception:
                mAP = None

    if mAP is None:
        print(f"Aviso: não foi possível extrair mAP do trial {trial.number}; retornando 0.0")
        mAP = 0.0

    return mAP


def run_optuna(trials: int, study_name: str, storage: str = None):
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    print('Melhores parâmetros:', study.best_params)
    print('Melhor valor (mAP):', study.best_value)
    return study


def final_train(best_params: dict):
    print('Iniciando treinamento final com melhores hiperparâmetros...')
    model = YOLO(WEIGHT_PRETRAIN)
    model.train(
        data=DATA_YAML,
        epochs=best_params.get('epochs', 300),
        batch=best_params.get('batch', 64),
        lr0=best_params.get('lr0', 1e-3),
        optimizer=best_params.get('optimizer', 'SGD'),
        weight_decay=best_params.get('weight_decay', 5e-4),
        name='final_best',
        project=SAVE_DIR,
        exist_ok=True,
        device=0,
        imgsz = 640
    )
    print('Treinamento final concluído. Checkpoints em', os.path.join(SAVE_DIR, 'final_best'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='Número de trials Optuna')
    parser.add_argument('--study-name', type=str, default='yolov11_optuna', help='Nome do estudo Optuna')
    parser.add_argument('--storage', type=str, default=None, help='SQLite storage URI (ex: sqlite:///optuna.db)')
    parser.add_argument('--final', action='store_true', help='Somente rodar o treino final usando best_params')
    args = parser.parse_args()

    if args.final:
        if args.storage is None:
            raise SystemExit('Para --final forneça --storage sqlite:///optuna.db')
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
        final_train(study.best_params)
    else:
        study = run_optuna(trials=args.trials, study_name=args.study_name, storage=args.storage)
        final_train(study.best_params)
