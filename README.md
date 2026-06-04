# Trabalho Final de Inteligência Artificial — Classificação de Tecidos Histopatológicos com PathMNIST

Projeto desenvolvido para o Trabalho Final da disciplina de Inteligência Artificial do curso de Sistemas de Informação da UniCatólica.

O objetivo é percorrer uma jornada prática que vai da implementação manual de uma rede neural em NumPy até o uso de arquiteturas modernas de visão computacional, transfer learning, fine-tuning, busca de hiperparâmetros e técnicas de explicabilidade aplicadas ao dataset PathMNIST.

> **Antes da entrega:** preencher os integrantes da equipe, o link do artigo científico e os resultados finais gerados após a única avaliação no conjunto de teste.

## Equipe

- Integrante 1: **Rafael Ângelo Meireles Azevedo**
- Integrante 2: **Luis Felipe Xavier Falcão**
- Integrante 3: **Victor Pinheiro de Lima**

## Artigo científico

- PDF do artigo: **adicionar link ou caminho em `report/relatorio.pdf`**

## Dataset

O projeto utiliza o **PathMNIST**, da coleção MedMNIST, com 9 classes de tecidos histopatológicos relacionados ao câncer colorretal:

1. Adiposo
2. Fundo
3. Debris
4. Linfócitos
5. Muco
6. Músculo liso
7. Mucosa normal
8. Estroma associado ao câncer
9. Epitélio adenocarcinomatoso

Na Etapa 1, a versão 28×28 é utilizada apenas para a implementação matemática da MLP em NumPy.

A partir da Etapa 2, são utilizadas imagens reais da versão oficial **224×224**, sem redimensionar imagens 28×28. Devido à limitação de memória do Google Colab gratuito, foi utilizado um subconjunto real da versão 224×224, preservando os splits oficiais:

| Split | Quantidade | Resolução |
|---|---:|---:|
| Treino | 3.000 | 224×224 |
| Validação | 800 | 224×224 |
| Teste | 1.000 | 224×224 |

O conjunto de teste é mantido isolado e deve ser usado uma única vez, apenas após todas as decisões de modelo e hiperparâmetros terem sido tomadas com base na validação.

## Objetivos atendidos

- Implementar uma MLP do zero utilizando apenas NumPy.
- Implementar forward propagation, backpropagation, Softmax, Cross-Entropy e SGD com Momentum.
- Validar os gradientes com Gradient Checking.
- Criar uma MLP equivalente em PyTorch, com comparação controlada em relação à versão NumPy.
- Construir uma CNN própria com pelo menos três blocos convolucionais.
- Avaliar MobileNetV2, ResNet18 e EfficientNet-B0 em Feature Extraction e Fine-tuning parcial.
- Avaliar uma arquitetura baseada em atenção com Swin Transformer.
- Executar um grid de 2 otimizadores × 3 learning rates.
- Implementar uma Etapa 5 com augmentation, scheduler, regularização, early stopping, checkpoint e logs.
- Aplicar Feature Maps e Grad-CAM em imagens de validação.
- Avaliar o melhor modelo no conjunto de teste apenas uma vez.
- Gerar matriz de confusão 9×9 e relatório de classificação.

## Estrutura do repositório

```text
trabalho-final-ia-pathmnist/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_carregar_pathmnist.ipynb
│   ├── 02_numpy_mlp.ipynb
│   ├── 03_pytorch_validation.ipynb
│   ├── 04_cnns_and_vit.ipynb
│   └── 05_criar_sample_224.ipynb
├── data/
│   └── arquivos .npz não versionados
├── experiments/
│   ├── results.csv
│   ├── grid_results.csv
│   ├── etapa5_training_log.csv
│   └── log_*.csv
├── checkpoints/
│   ├── mobilenetv2_etapa5_best.pth
│   ├── modelo_final_etapa5.pth
│   └── modelo_final_etapa5_metadata.json
├── figures/
│   ├── comparacao_acuracia.png
│   ├── feature_maps.png
│   ├── gradcam_acertos.png
│   ├── gradcam_erros.png
│   ├── gradcam_acerto_por_sorte.png
│   ├── etapa5_curvas_loss.png
│   ├── etapa5_curvas_acuracia.png
│   └── matriz_confusao.png
└── report/
    └── relatorio.pdf
```

## Notebooks

### `01_carregar_pathmnist.ipynb`

Carregamento e visualização inicial do PathMNIST.

### `02_numpy_mlp.ipynb`

Implementação manual da MLP com NumPy:

- Preparação dos dados.
- One-hot encoding.
- ReLU e derivada.
- Softmax numericamente estável.
- Cross-Entropy.
- Forward propagation.
- Backpropagation.
- SGD com Momentum.
- Curvas de loss e acurácia.
- Gradient Checking com diferença relativa esperada menor que `10⁻⁵`.

### `03_pytorch_validation.ipynb`

Implementação da MLP equivalente em PyTorch:

- Mesmos dados, ordem de achatamento, seed, pesos iniciais e hiperparâmetros da versão NumPy.
- Comparação controlada após 20 épocas.
- Verificação automática da diferença de acurácia em pontos percentuais.

### `04_cnns_and_vit.ipynb`

Notebook principal das etapas avançadas:

- Carregamento do sample real 224×224.
- CNN própria.
- MobileNetV2, ResNet18 e EfficientNet-B0 em Feature Extraction.
- MobileNetV2, ResNet18 e EfficientNet-B0 em Fine-tuning parcial.
- Swin Transformer.
- Grid de SGD com Momentum e AdamW usando learning rates `1e-2`, `1e-3` e `1e-4`.
- Etapa 5 com data augmentation moderado, label smoothing, weight decay, scheduler cosseno e early stopping.
- Salvamento de resultados, logs e checkpoints.
- Grad-CAM em validação: 5 acertos com alta confiança, 5 erros grosseiros e 1 candidato a acerto por sorte.
- Feature Maps.
- Única avaliação final no conjunto de teste.
- Matriz de confusão 9×9 e relatório de classificação.

### `05_criar_sample_224.ipynb`

Criação dos subconjuntos reais da versão oficial 224×224, preservando os splits oficiais de treino, validação e teste.

## Modelos avaliados

| Modelo | Estratégia |
|---|---|
| MLP NumPy | Implementação manual |
| MLP PyTorch | Treino do zero e equivalência controlada |
| CNN própria | Treino do zero |
| MobileNetV2 | Feature Extraction e Fine-tuning parcial |
| ResNet18 | Feature Extraction e Fine-tuning parcial |
| EfficientNet-B0 | Feature Extraction e Fine-tuning parcial |
| Swin-T | Feature Extraction |
| MobileNetV2 Etapa 5 | Fine-tuning parcial com otimização avançada |

## Grid de hiperparâmetros

O grid compara duas estratégias de otimização em condições controladas:

| Otimizador | Learning Rates |
|---|---|
| SGD com Momentum | `1e-2`, `1e-3`, `1e-4` |
| AdamW | `1e-2`, `1e-3`, `1e-4` |

Cada combinação começa com uma nova MobileNetV2 pré-treinada, usa a mesma seed, os mesmos dados e o mesmo número de épocas. A melhor configuração é escolhida exclusivamente pela validação.

## Etapa 5 — Desafio Final

A Etapa 5 utiliza a melhor configuração do grid e combina:

- Data augmentation moderado.
- Fine-tuning parcial da MobileNetV2.
- Label smoothing.
- Weight decay.
- Scheduler `CosineAnnealingLR`.
- Early stopping.
- Salvamento do melhor checkpoint pela menor loss de validação.
- Logs por época em CSV.

As transformações geométricas e de cor foram mantidas moderadas para evitar distorções excessivas nas características histológicas.

## Explicabilidade

### Grad-CAM

O Grad-CAM é aplicado exclusivamente em imagens do conjunto de validação. A seleção inclui:

- 5 acertos com maior confiança.
- 5 erros grosseiros com maior confiança na classe errada.
- 1 candidato a acerto por sorte, identificado por uma heurística de atenção concentrada nas bordas.

Os mapas devem ser analisados visualmente para verificar se o modelo concentra atenção em regiões teciduais relevantes ou em fundo, bordas e artefatos.

### Feature Maps

Os Feature Maps das primeiras camadas convolucionais permitem observar ativações associadas a bordas, texturas, contrastes e variações de intensidade. Essas representações simples são combinadas nas camadas mais profundas para formar padrões mais complexos relacionados às classes histopatológicas.

## Resultados e artefatos

Os resultados devem ser consultados nos arquivos gerados pelo notebook:

- `experiments/results.csv`: resumo consolidado de modelos, grid e Etapa 5.
- `experiments/grid_results.csv`: resultados completos do grid.
- `experiments/etapa5_training_log.csv`: histórico detalhado da Etapa 5.
- `experiments/log_*.csv`: logs dos demais modelos.
- `checkpoints/modelo_final_etapa5.pth`: checkpoint final selecionado pela validação.
- `checkpoints/modelo_final_etapa5_metadata.json`: metadados do modelo final.

### Resultado final no teste

> **Preencher somente após executar a seção final do teste uma única vez. Não alterar o modelo ou os hiperparâmetros após observar esse resultado.**

| Modelo Final | Acurácia no Teste | Loss no Teste |
|---|---:|---:|
| Preencher após a avaliação final | Preencher | Preencher |

## Reprodutibilidade

A seed utilizada nos notebooks é:

```text
SEED = 42
```

As seeds são fixadas para:

- `random`
- `numpy`
- `torch`
- `torch.cuda`, quando disponível

Os experimentos foram executados nos seguintes ambientes:

### Ambiente local

- Sistema operacional: Windows
- Editor: Visual Studio Code
- Python: 3.12
- Processador: Intel Core i5 de 10ª geração
- Memória RAM: 8 GB

### Ambiente em nuvem

- Google Colab
- GPU: NVIDIA T4
- VRAM: **preencher conforme o ambiente exibido pelo Colab**

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/rafaelxde/trabalho-final-ia-pathmnist.git
cd trabalho-final-ia-pathmnist
```

### 2. Criar e ativar um ambiente virtual

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Caso o PowerShell bloqueie a ativação:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Linux ou macOS:

```bash
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Executar os notebooks

Ordem recomendada:

1. `01_carregar_pathmnist.ipynb`
2. `02_numpy_mlp.ipynb`
3. `03_pytorch_validation.ipynb`
4. `05_criar_sample_224.ipynb`
5. `04_cnns_and_vit.ipynb`

No notebook `04_cnns_and_vit.ipynb`, execute:

1. Modelos e fine-tunings.
2. Grid de hiperparâmetros.
3. Etapa 5.
4. Salvamento de resultados, logs e checkpoints.
5. Grad-CAM e Feature Maps.
6. Avaliação final no teste, apenas uma vez.

## Observação sobre checkpoints

Os checkpoints da MobileNetV2 normalmente possuem tamanho compatível com o limite do GitHub. Caso algum arquivo ultrapasse o limite permitido, disponibilize um link de download no README e mantenha os metadados do modelo no repositório.

## Uso de Inteligência Artificial Generativa

Durante o desenvolvimento, o ChatGPT foi utilizado como apoio para:

- Organização das etapas do projeto.
- Explicação de conceitos.
- Auxílio na depuração de erros.
- Estruturação e revisão dos notebooks.
- Revisão da documentação.
- Apoio na implementação de controles de reprodutibilidade.

Todo o código foi executado, analisado e adaptado pela equipe. A equipe deve ser capaz de explicar e modificar qualquer trecho apresentado.

## Limitações

- Os experimentos avançados utilizam um subconjunto real da versão 224×224 devido à limitação de memória do ambiente gratuito.
- O desempenho pode variar em execuções futuras, apesar da fixação de seeds.
- Grad-CAM fornece evidências visuais úteis, mas não garante validade clínica.
- O modelo deve ser interpretado como ferramenta de apoio e não como substituto da avaliação médica especializada.
