# Trabalho Final de Inteligência Artificial — Classificação de Imagens Médicas com PathMNIST

## Descrição do Projeto

Este projeto tem como objetivo desenvolver, treinar, comparar e interpretar modelos de Inteligência Artificial para classificação de imagens médicas do dataset **PathMNIST**, pertencente à coleção **MedMNIST**.

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

O dataset utilizado foi o **PathMNIST**, da biblioteca **MedMNIST**.

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

Durante as etapas iniciais em NumPy e PyTorch, foi utilizada a versão **28x28** do dataset para reduzir o custo computacional e focar na implementação matemática da rede neural.

Nas etapas avançadas, após orientação do professor, foi utilizado um **subconjunto real da versão oficial 224x224 do PathMNIST**, preservando os splits oficiais de treino, validação e teste. Essa decisão foi tomada porque o carregamento completo da versão 224x224 excedeu a memória RAM disponível no Google Colab gratuito.

O subconjunto utilizado foi:

| Split     | Quantidade de imagens | Resolução |
| --------- | --------------------: | --------- |
| Treino    |                  3000 | 224x224   |
| Validação |                   800 | 224x224   |
| Teste     |                  1000 | 224x224   |

Dessa forma, os experimentos avançados foram realizados com imagens reais **224x224**, sem redimensionar imagens 28x28 para essa etapa.

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

* Criação e carregamento do sample real 224x224
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

### Resultados de Validação com Sample Real 224x224

| Modelo                  | Estratégia          | Melhor Acurácia de Validação |
| ----------------------- | ------------------- | ---------------------------: |
| CNN própria             | Treino do zero      |                       48,63% |
| MobileNetV2             | Feature Extraction  |                       93,63% |
| ResNet18                | Feature Extraction  |                       92,37% |
| EfficientNet-B0         | Feature Extraction  |                       93,37% |
| Swin-T                  | Feature Extraction  |                       94,25% |
| MobileNetV2 Fine-tuning | Fine-tuning parcial |                       95,50% |

O melhor modelo durante a validação foi a **MobileNetV2 com fine-tuning parcial**, atingindo aproximadamente **95,50% de acurácia na validação**.

A CNN própria apresentou desempenho inferior por ter sido treinada do zero com um subconjunto reduzido de dados. Já os modelos pré-treinados apresentaram resultados superiores, demonstrando a vantagem do **Transfer Learning** em tarefas de classificação de imagens médicas.

### Resultado Final no Teste

O conjunto de teste foi utilizado apenas uma vez, após a escolha do melhor modelo com base na validação.

| Modelo Final            | Acurácia no Teste | Loss no Teste |
| ----------------------- | ----------------: | ------------: |
| MobileNetV2 Fine-tuning |            92,30% |        0.2984 |

O resultado final no teste foi de **92,30% de acurácia**, utilizando o subconjunto real 224x224 do PathMNIST.

### Relatório por Classe

O modelo apresentou bom desempenho geral no conjunto de teste. As classes adiposo, fundo, linfócitos, muco, mucosa normal e epitélio adenocarcinomatoso obtiveram altos valores de F1-score.

A principal dificuldade ocorreu na classe **estroma associado ao câncer**, que apresentou recall reduzido. Isso indica que muitos exemplos reais dessa classe foram classificados como outras categorias, possivelmente devido à semelhança visual com outros tecidos histopatológicos.

## Explicabilidade

Foram utilizadas técnicas de explicabilidade para analisar o comportamento do modelo final.

### Grad-CAM

O Grad-CAM foi aplicado utilizando o modelo final, a **MobileNetV2 com fine-tuning parcial treinada no sample real 224x224**. A técnica permitiu visualizar as regiões das imagens que mais influenciaram as decisões do modelo.

Foram analisados exemplos classificados corretamente e incorretamente. Nos acertos, os mapas de calor indicaram regiões relevantes para a decisão da rede. Nos erros, os mapas ajudaram a observar possíveis regiões de confusão, especialmente em classes visualmente semelhantes.

Essa análise é importante porque, em problemas médicos, a acurácia isolada não é suficiente. Também é necessário observar se o modelo está tomando decisões com base em regiões visualmente coerentes da imagem.

### Feature Maps

Também foram visualizados Feature Maps das primeiras camadas convolucionais da MobileNetV2. Esses mapas mostraram ativações associadas a padrões visuais simples, como bordas, texturas, regiões de contraste e variações de intensidade.

Essas ativações iniciais são combinadas nas camadas mais profundas para formar representações mais complexas relacionadas às classes histopatológicas do PathMNIST.

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
* Utilizado para treinamento dos modelos pré-treinados, Swin Transformer, Grad-CAM, Feature Maps e avaliação final

## Observação sobre Limitação Computacional

Durante os experimentos, o carregamento completo do **PathMNIST 224x224** excedeu a memória RAM disponível no Google Colab gratuito.

Seguindo orientação do professor, foi criado um **subconjunto real da versão oficial 224x224 do PathMNIST**, preservando os splits oficiais de treino, validação e teste.

Foram utilizadas:

* 3000 imagens de treino
* 800 imagens de validação
* 1000 imagens de teste

Dessa forma, os experimentos avançados foram realizados com imagens reais 224x224, sem utilizar imagens 28x28 redimensionadas para essa etapa.

Os arquivos `.npz` do sample não foram adicionados ao GitHub por serem arquivos de dados. Eles foram utilizados apenas para execução dos experimentos no ambiente do Google Colab.

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

Nas etapas avançadas, os modelos pré-treinados superaram significativamente a CNN própria, mostrando a vantagem do Transfer Learning. O melhor desempenho foi obtido pela **MobileNetV2 com fine-tuning parcial**, alcançando **95,50% de acurácia na validação** e **92,30% de acurácia no conjunto de teste**.

A análise com Grad-CAM e Feature Maps mostrou que a explicabilidade é essencial em aplicações médicas, pois permite observar se o modelo está tomando decisões com base em regiões visualmente coerentes. Apesar do bom desempenho, o modelo deve ser visto como ferramenta de apoio, e não como substituto de avaliação médica especializada.
