# Configuração do pip

## Pré-requisitos para Treinamento do YOLOv5

### 1. Configuração do Ambiente Python
1. Certifique-se de ter o Python 3.8 ou superior instalado.
2. Crie um ambiente virtual local (venv):
   ```bash
   python3 -m venv venv
   ```
3. Ative o ambiente virtual:
   ```bash
   source venv/bin/activate
   ```
4. Instale os módulos necessários:
   ```bash
   pip install -r yolov5/requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
5. Configuração e execução do Jupyter
  ```bash
   pip install jupyter matplotlib numpy pandas scipy scikit-learn
   pip install ipykernel
   python -m ipykernel install --user --name="venv" --display-name="TI6"
   jupyter notebook
  ```


### 2. Requisitos de Hardware
- GPU compatível com CUDA para aceleração de treinamento.
- Verifique a existência do driver NVIDIA:
  ```bash
  nvidia-smi
  ```
- Instale o driver NVIDIA (exemplo para driver 550):
  ```bash
  sudo apt install nvidia-driver-550
  ```

### 3. Preparação do Dataset
- Estrutura esperada para os datasets:
  ```
  datasets/
    coco128/
      images/
        train2017/
      labels/
        train2017/
    CSDD_det/
      images/
        train2017/
        val2017/
      labels/
        train2017/
        val2017/
  ```
- Certifique-se de que os arquivos de imagem e rótulo estão organizados corretamente.

### 4. Execução do Treinamento
1. Navegue até o diretório `yolov5`:
   ```bash
   cd yolov5
   ```
2. Execute o script de treinamento:
   ```bash
   python train.py --data data/coco128.yaml --weights yolov5s.pt --epochs 100
   ```

### 5. Dependências Opcionais
- Para funcionalidades adicionais, instale os seguintes pacotes:
  - Logging: `tensorboard`, `clearml`
  - Exportação: `onnx`, `tensorflow`, `openvino-dev`
  - Plotagem: `seaborn`, `pandas`

Consulte o arquivo `requirements.txt` para mais detalhes sobre as dependências.