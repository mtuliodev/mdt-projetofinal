# Predicting Brazilian Court Decisions — Guia de Reexecução

Projeto de Processamento de Linguagem Natural baseado no artigo:

> Lage-Freitas, A. et al. (2022). Predicting Brazilian Court Decisions.
> PeerJ Computer Science. DOI: 10.7717/peerj-cs.904

O projeto compara dois pipelines de classificacao de decisoes judiciais usando o mesmo
dataset e a mesma etapa de limpeza do artigo original, variando apenas a vetorizacao e
os modelos utilizados.

---

## Estrutura dos arquivos

```
tfidf_pipeline_court_decisions.ipynb      Pipeline A: TF-IDF + modelos classicos
transformers_pipeline_court_decisions.ipynb   Pipeline B: Transformers (fine-tuning)
```

---

## Dataset

O dataset e carregado automaticamente pelo Hugging Face Datasets.
Nenhum download manual e necessario.

- Fonte: `joelniklaus/brazilian_court_decisions`
- Conteudo: 4.043 acordaos do Tribunal de Justica de Alagoas (TJAL)
- Rotulos: `yes` (recurso provido), `no` (nao provido), `partial` (parcialmente provido)
- Os rotulos foram extraidos automaticamente do texto das decisoes, sem anotacao manual

---

## Requisitos de ambiente

### Versao do Python

Python 3.9 ou superior. Testado com Python 3.14.

### Execucao recomendada

Os notebooks foram desenvolvidos para execucao no Google Colab.
O notebook de Transformers exige GPU para tempo de execucao aceitavel.

Para ativar GPU no Colab:
`Menu > Runtime > Change runtime type > Hardware accelerator > GPU (T4)`

E possivel executar localmente, mas o notebook de Transformers pode levar
varias horas em CPU. O notebook TF-IDF funciona bem em qualquer maquina.

---

## Notebook A: TF-IDF + Modelos Classicos

**Arquivo:** `tfidf_pipeline_court_decisions.ipynb`

### Instalacao de dependencias

Execute a primeira celula do notebook, ou instale manualmente:

```bash
pip install datasets nltk scikit-learn xgboost pandas numpy unidecode seaborn matplotlib
```

Alem disso, o notebook baixa automaticamente tres corpora do NLTK na execucao:
- `rslp` (stemmer para portugues)
- `stopwords`
- `punkt`

### Sequencia de execucao

Execute as celulas em ordem. Cada secao depende da anterior.

| Secao | O que faz | Tempo estimado |
|-------|-----------|----------------|
| 0. Instalacao | Instala pacotes | 1-2 min |
| 1. Imports | Carrega bibliotecas | < 30s |
| 2. Carregamento | Baixa dataset do Hugging Face | 1-3 min (depende da conexao) |
| 3. Limpeza | Remove duplicatas e rotulos invalidos | < 30s |
| 4. Pre-processamento | Aplica stopwords + stemmer RSLP | 2-5 min |
| 5. Vetorizacao | Gera matriz TF-IDF (10k features, bigramas) | < 1 min |
| 6. Divisao | Separa treino/teste (80/20, seed=42) | < 5s |
| 7. Funcao de avaliacao | Define funcao usada por todos os modelos | < 5s |
| 8. Regressao Logistica | Treina e avalia (inclui CV 5-fold) | 30s-2 min |
| 9. Modelos de referencia | SVM, XGBoost, Random Forest, Naive Bayes | 2-10 min |
| 10. Visualizacao | Gera 4 graficos comparativos | 1-2 min |
| 11. Conclusao | Leitura apenas | - |

### Saidas geradas

- `results_tfidf.csv`: metricas de todos os modelos (necessario para o notebook B)
- `f1_macro_tfidf.png`: grafico de barras F1-macro
- `f1_por_classe_tfidf.png`: F1 por classe (yes / partial / no)
- `matrizes_confusao_tfidf.png`: matrizes de confusao de todos os modelos
- `radar_tfidf.png`: grafico radar multi-metrica

### Modelos e resultados obtidos

| Modelo | F1-macro | Acuracia | Tempo de treino |
|--------|----------|----------|-----------------|
| XGBoost | 70.3% | 77.8% | 70s |
| SVM (LinearSVC) | 69.8% | 77.5% | < 1s |
| Regressao Logistica | 65.1% | 74.7% | 2s |
| Random Forest | 54.7% | 70.2% | 1s |
| Naive Bayes | 54.2% | 63.4% | < 1s |

Nota: o artigo original reportou F1-macro de aproximadamente 80% para XGBoost
usando validacao cruzada de 5 folds. Os valores acima usam split simples 80/20,
o que tende a resultar em F1 ligeiramente menor porem e mais rapido de executar.

---

## Notebook B: Transformers

**Arquivo:** `transformers_pipeline_court_decisions.ipynb`

### Instalacao de dependencias

As celulas de instalacao estao comentadas no notebook (os pacotes ja estavam
instalados no ambiente de execucao original). Descomente e execute a celula 0,
ou instale manualmente:

```bash
pip install datasets nltk scikit-learn pandas numpy seaborn matplotlib
pip install transformers torch sentence-transformers accelerate
```

Em ambientes com GPU CUDA, o PyTorch sera automaticamente utilizado.
Em CPU, o fine-tuning funciona mas e significativamente mais lento.

### Sequencia de execucao

Execute o notebook A antes do notebook B.
O notebook B le o arquivo `results_tfidf.csv` gerado pelo notebook A
para produzir o grafico comparativo final (secao 13).

| Secao | O que faz | Tempo estimado (GPU T4) |
|-------|-----------|------------------------|
| 0. Instalacao | Descomente e instale | 2-5 min |
| 1. Imports | Carrega bibliotecas | < 1 min |
| 2. Carregamento | Mesmo dataset do notebook A | 1-3 min |
| 3. Limpeza | Identica ao notebook A | < 30s |
| 4. Pre-processamento | Limpeza leve (sem stemming) | < 30s |
| 5. Divisao | Mesma seed=42 do notebook A | < 5s |
| 6. Funcao de avaliacao | Define funcao de metricas | < 5s |
| 7. Infraestrutura BERT | Define CourtDataset e funcao de treino | < 5s |
| 8. DistilBERT Portugues | Fine-tuning 3 epocas | ~60 min (GPU) |
| 9. Legal-BERT PT-BR | Fine-tuning 3 epocas | ~1h 47min (GPU) |
| 10. Sentence-BERT | Embeddings + Regressao Logistica | ~18 min |
| 11. BERTimbau (opcional) | Fine-tuning 3 epocas | ~60 min (GPU) |
| 12. Visualizacao | 4 graficos comparativos | 1-2 min |
| 13. Comparacao cruzada | Requer results_tfidf.csv do notebook A | < 1 min |
| 14. Conclusao | Leitura apenas | - |

### Por que o pre-processamento e diferente do notebook A

O notebook TF-IDF aplica stemming e remove stopwords porque o TF-IDF
conta palavras exatas: "provido" e "provimento" sao tokens diferentes sem
stemming, o que dilui o sinal. Transformers usam tokenizacao de subpalavras
e mecanismo de atencao contextual: eles ja capturam a relacao entre "provido"
e "provimento". Remover stopwords ou aplicar stemming antes de um Transformer
pode destruir informacao de contexto relevante (por exemplo, remover "nao"
inverte o significado de uma sentenca).

### Saidas geradas

- `results_transformers.csv`: metricas de todos os modelos Transformer
- `f1_macro_transformers.png`: grafico de barras F1-macro
- `f1_por_classe_transformers.png`: F1 por classe
- `matrizes_confusao_transformers.png`: matrizes de confusao
- `radar_transformers.png`: grafico radar multi-metrica
- `comparacao_geral_tfidf_vs_transformers.png`: ranking geral dos dois notebooks

### Modelos e resultados obtidos

| Modelo | F1-macro | Acuracia | Tempo de treino |
|--------|----------|----------|-----------------|
| Legal-BERT PT-BR | 60.9% | 70.6% | 1h 47min |
| Sentence-BERT Juridico | 50.7% | 63.4% | 18min |
| DistilBERT Portugues | 48.7% | 61.8% | 61min |
| BERTimbau Base | nao executado | - | ~60min |

Os modelos sao carregados automaticamente do Hugging Face na primeira execucao.
Espaco em disco necessario (download unico por modelo):

| Modelo | Tamanho aproximado |
|--------|--------------------|
| adalbertojunior/distilbert-portuguese-cased | 250 MB |
| dominguesm/legal-bert-base-cased-ptbr | 440 MB |
| rufimelo/Legal-BERTimbau-sts-base-scaled | 440 MB |
| neuralmind/bert-base-portuguese-cased | 440 MB |

---

## Ordem recomendada de execucao

1. Execute o notebook A (TF-IDF) completamente.
   Certifique-se de que `results_tfidf.csv` foi salvo no mesmo diretorio.

2. Execute o notebook B (Transformers) completamente.
   A secao 13 ira detectar automaticamente o `results_tfidf.csv`
   e gerar o grafico de comparacao cruzada entre os dois pipelines.

---

## Reproducibilidade

Ambos os notebooks usam `random_state=42` na divisao treino/teste.
O conjunto de teste e identico nos dois notebooks, o que garante
comparabilidade direta dos resultados.

A limpeza de dados (remocao de duplicatas, valores nulos e rotulos invalidos)
e identica ao codigo original do artigo de referencia.

Os resultados podem variar ligeiramente dependendo de:
- Versao das bibliotecas instaladas
- Hardware utilizado (CPU vs GPU, modelo de GPU)
- Versao dos modelos baixados do Hugging Face

---

## Solucao de problemas comuns

**Erro ao carregar o dataset:**
```
ValueError: BuilderConfig 'judgment' not found
```
Verifique se a chamada esta sem o segundo argumento:
```python
raw = load_dataset('joelniklaus/brazilian_court_decisions')
```

**GPU nao detectada no Colab:**
Acesse `Runtime > Change runtime type` e selecione `T4 GPU` antes de iniciar a execucao.
Reinicie o ambiente apos a mudanca.

**Erro de memoria (OOM) durante fine-tuning:**
Reduza o `batch_size` na funcao `treinar_avaliar_bert`:
```python
y_pred = treinar_avaliar_bert(..., batch_size=8)
```

**Download de modelo muito lento ou falha:**
Os modelos sao cacheados localmente apos o primeiro download.
Em caso de falha de rede, execute a celula novamente.

**`results_tfidf.csv` nao encontrado na secao 13 do notebook B:**
Execute o notebook A ate o final e certifique-se de que o arquivo
`results_tfidf.csv` foi salvo no mesmo diretorio do notebook B.

---

## Referencia

Lage-Freitas, A., Allende-Cid, H., Santana, O., & Oliveira-Lage, L. (2022).
Predicting Brazilian Court Decisions.
PeerJ Computer Science, 8, e904.
https://doi.org/10.7717/peerj-cs.904

Dataset: https://huggingface.co/datasets/joelniklaus/brazilian_court_decisions
