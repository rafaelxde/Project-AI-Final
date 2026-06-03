# Trabalho Final de Inteligência Artificial — Classificação de Imagens Médicas com PathMNIST

## Descrição do Projeto

Este projeto tem como objetivo desenvolver, treinar, comparar e interpretar modelos de Inteligência Artificial para classificação de imagens médicas do dataset **PathMNIST**, pertencente à coleção MedMNIST.

O PathMNIST contém imagens histopatológicas relacionadas a tecidos de câncer colorretal, organizadas em **9 classes**. A proposta do trabalho foi seguir uma evolução prática de modelos de aprendizado profundo, começando por uma rede neural implementada manualmente em NumPy e avançando até modelos convolucionais pré-treinados, Transformer visual e técnicas de explicabilidade.

## Objetivos

* Implementar uma MLP do zero utilizando apenas NumPy.
* Implementar forward propagation, backpropagation, softmax, entropia cruzada e SGD com Momentum.
* Validar os gradientes com Gradient Checking.
* Recriar uma MLP equivalente utilizando PyTorch.
* Treinar uma CNN própria.
* Avaliar modelos pré-treinados por Transfer Learning.
* Testar uma arquitetura baseada em atenção, utilizando Swin Transformer.
* Aplicar fine-tuning parcial no melhor modelo.
* Utilizar técnicas de explicabilidade, como Feature Maps e Grad-CAM.
* Avaliar o melhor modelo no conjunto de teste apenas uma vez.
* Gerar matriz de confusão e relatório de classificação.

## Dataset

O dataset utilizado foi o **PathMNIST**, da biblioteca MedMNIST.

As imagens pertencem a 9 classes:

1. Adiposo
2. Fundo
3. Debris
4. Linfócitos
5. Muco
6. Músculo liso
7. Mucosa normal
8. Estroma associado ao câncer
9. Epitélio adenocarcinomatoso

Durante as etapas iniciais em NumPy e PyTorch, foi utilizada a versão 28x28 do dataset para reduzir o custo computacional e focar na implementação matemática.

Nas etapas com CNNs, modelos pré-treinados e Swin Transformer, as imagens foram redimensionadas para 224x224 com `torchvision.transforms.Resize`, mantendo compatibilidade com arquiteturas pré-treinadas.

## Estrutura do Projeto

```txt
trabalho-final-ia-pathmnist/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_carregar_pathmnist.ipynb
│   ├── 02_numpy_mlp.ipynb
│   ├── 03_pytorch_validation.ipynb
│   └── 04_cnns_and_vit.ipynb
├── data/
├── experiments/
├── figures/
├── report/
└── src/
```

## Notebooks

### `01_carregar_pathmnist.ipynb`

Notebook inicial usado para carregar e visualizar o dataset PathMNIST.

### `02_numpy_mlp.ipynb`

Implementação da MLP do zero com NumPy.

Contém:

* Preparação dos dados
* One-hot encoding
* ReLU e derivada da ReLU
* Softmax
* Cross-Entropy
* Forward propagation
* Backpropagation
* SGD com Momentum
* Treinamento
* Curvas de loss e acurácia
* Gradient Checking

### `03_pytorch_validation.ipynb`

Implementação de uma MLP equivalente utilizando PyTorch.

Contém:

* DataLoaders
* Modelo MLP em PyTorch
* Treinamento por 20 épocas
* Comparação com a versão NumPy

### `04_cnns_and_vit.ipynb`

Notebook principal das etapas avançadas.

Contém:

* CNN própria
* MobileNetV2 pré-treinada
* ResNet18 pré-treinada
* EfficientNet-B0 pré-treinada
* Swin Transformer
* Fine-tuning parcial da MobileNetV2
* Tabela comparativa dos modelos
* Gráficos de comparação
* Grad-CAM
* Feature Maps
* Avaliação final no conjunto de teste
* Matriz de confusão
* Relatório de classificação

## Modelos Avaliados

Foram avaliados os seguintes modelos:

| Modelo                  | Estratégia           |     Otimizador | Learning Rate |
| ----------------------- | -------------------- | -------------: | ------------: |
| MLP NumPy               | Implementação manual | SGD + Momentum |          0.01 |
| MLP PyTorch             | Treino do zero       | SGD + Momentum |          0.01 |
| CNN própria             | Treino do zero       |           Adam |         0.001 |
| MobileNetV2             | Feature Extraction   |          AdamW |         0.001 |
| ResNet18                | Feature Extraction   |          AdamW |         0.001 |
| EfficientNet-B0         | Feature Extraction   |          AdamW |         0.001 |
| Swin-T                  | Feature Extraction   |          AdamW |         0.001 |
| MobileNetV2 Fine-tuning | Fine-tuning parcial  |          AdamW |        0.0001 |

## Resultados Principais

### Resultados de Validação

| Modelo                  | Estratégia          |              Acurácia de Validação |
| ----------------------- | ------------------- | ---------------------------------: |
| CNN própria             | Treino do zero      |                             43,12% |
| MobileNetV2             | Feature Extraction  |                             88,00% |
| ResNet18                | Feature Extraction  |                             83,37% |
| EfficientNet-B0         | Feature Extraction  |                             85,50% |
| Swin-T                  | Feature Extraction  |                             85,88% |
| MobileNetV2 Fine-tuning | Fine-tuning parcial | 89,88% final / 91,63% melhor época |

O melhor modelo durante a validação foi a **MobileNetV2 com fine-tuning parcial**, atingindo pico de aproximadamente **91,63% de acurácia na validação**.

### Resultado Final no Teste

O conjunto de teste foi utilizado apenas uma vez, após a escolha do melhor modelo com base na validação.

| Modelo Final            | Acurácia no Teste | Loss no Teste |
| ----------------------- | ----------------: | ------------: |
| MobileNetV2 Fine-tuning |            87,80% |        0.3991 |

## Explicabilidade

Foram utilizadas técnicas de explicabilidade para analisar o comportamento do modelo final.

### Grad-CAM

O Grad-CAM foi aplicado em exemplos classificados corretamente e incorretamente. A técnica permitiu visualizar as regiões das imagens que mais influenciaram as decisões do modelo.

A análise mostrou que, em vários acertos, o modelo concentrou atenção em regiões visualmente relevantes. Já nos erros, principalmente em exemplos com alta confiança, os mapas indicaram possíveis confusões entre regiões visualmente semelhantes.

### Feature Maps

Também foram visualizados Feature Maps das primeiras camadas convolucionais da MobileNetV2. Esses mapas mostraram ativações associadas a padrões visuais simples, como bordas, texturas e variações de intensidade.

## Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/rafaelxde/trabalho-final-ia-pathmnist.git
cd trabalho-final-ia-pathmnist
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv
```

### 3. Ativar ambiente virtual

No Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Caso apareça erro de permissão:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 4. Instalar dependências

```bash
pip install -r requirements.txt
```

### 5. Abrir os notebooks

Os notebooks podem ser abertos no VS Code, Jupyter Notebook ou Google Colab.

Para as etapas mais pesadas, recomenda-se utilizar o Google Colab com GPU:

```txt
Ambiente de execução > Alterar tipo de ambiente de execução > GPU
```

## Ambiente de Execução

### Ambiente local

* Sistema operacional: Windows
* Editor: Visual Studio Code
* Python: 3.12
* Hardware local: notebook com processador Intel Core i5 de 10ª geração e 8 GB de RAM

### Ambiente em nuvem

* Google Colab
* GPU: NVIDIA T4
* Utilizado para treinamento dos modelos pré-treinados, Swin Transformer, Grad-CAM e avaliação final

## Observação sobre Limitação Computacional

Durante os experimentos no Google Colab gratuito, o carregamento completo do PathMNIST em resolução 224x224 excedeu a memória RAM disponível. Por isso, para os experimentos com CNNs e modelos pré-treinados, foi utilizada a versão 28x28 do PathMNIST redimensionada para 224x224 com `torchvision.transforms.Resize`.

Essa decisão permitiu manter a compatibilidade com os modelos pré-treinados e viabilizar os experimentos dentro das limitações computacionais disponíveis.

## Uso de Inteligência Artificial Generativa

Durante o desenvolvimento, foi utilizado o ChatGPT como apoio para:

* Organização das etapas do projeto
* Explicação de conceitos
* Auxílio na depuração de erros
* Estruturação dos notebooks
* Revisão de textos explicativos
* Apoio na documentação do projeto

Todo o código foi executado, analisado e adaptado pela equipe.

## Conclusão

O projeto demonstrou a evolução prática de modelos de Inteligência Artificial aplicados à classificação de imagens médicas. A implementação manual em NumPy permitiu compreender os fundamentos matemáticos de uma rede neural, enquanto a versão em PyTorch validou a migração para um framework profissional.

Os modelos pré-treinados superaram significativamente a CNN própria, mostrando a vantagem do Transfer Learning. O melhor desempenho foi obtido pela MobileNetV2 com fine-tuning parcial, alcançando 87,80% de acurácia no conjunto de teste.

A análise com Grad-CAM e Feature Maps mostrou que a explicabilidade é essencial em aplicações médicas, pois permite observar se o modelo está tomando decisões com base em regiões visualmente coerentes. Apesar do bom desempenho, o modelo deve ser visto como ferramenta de apoio, e não como substituto de avaliação especializada.