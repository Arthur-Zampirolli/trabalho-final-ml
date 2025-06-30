# Objetivo
Este repositório tem como objetivo realizar a análise de dados e treinamento de um modelo preditor de valor de venda de jogadores de futebol. 
# Instruções de uso
1.  Instale o DBeaver  
2. Baixe as bases com o comando:  
  ```#!/bin/bash curl -L -o {YOUR_REPO_PATH}player-scores.zip\ https://www.kaggle.com/api/v1/datasets/download/davidcariboo/player-scores```  
  (sendo {YOUR_REPO_PATH} o caminho do seu repositório)  
3.  Crie um banco sqlite com um nome qualquer. Ex: database.sqlite  
4.  Rode o script de create das tabelas em ordem crescente.  
5.  Importe os dados baixados em [2] pelo próprio dbeaver. Essa operação pode demorar um tempo, vai depender da velocidade do disco rígido da sua máquina.

# Football Transfer Fee Prediction - Machine Learning Project

This repository contains code and data for predicting football transfer fees using various machine learning models, including SVR, Gradient Boosting, and MLP. The project includes feature engineering, model training, and evaluation scripts.

## Project Structure

- `training/` - Contains model training scripts (e.g., `svr.py`, `xgboost.py`, `mlp.py`, `decision_tree.py`).
- `features_engineering_manual/` - Manual feature engineering scripts and outputs.
- `results/` - Stores results, metrics, and plots from model runs.
- `requirements.txt` - Python dependencies.

## Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd trabalho-final-ml
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Running Training Scripts

All training scripts are located in the `training/` directory. You can run any model script using the following command:

```sh
python training/<model>.py --target-column transfer_fee --output-dir results/training/<model>/
```

Replace `<model>` with one of: `svr`, `xgboost`, `mlp`, or `decision_tree`.

### Example: Run SVR Training

```sh
python training/svr.py --target-column transfer_fee --output-dir results/training/svr/
```

### Example: Run Gradient Boosting Training

```sh
python training/xgboost.py --target-column transfer_fee --output-dir results/training/xgboost/
```

### Example: Run MLP Training

```sh
python training/mlp.py --target-column transfer_fee --output-dir results/training/mlp/
```

### Example: Run Decision Tree Training

```sh
python training/decision_tree.py --target-column transfer_fee --output-dir results/training/decision_tree/
```

## Notes
- Make sure the feature-engineered CSV files exist in the expected locations (see scripts for paths).
- Output metrics and plots will be saved in the specified `--output-dir`.

## License
MIT License