# Pre-processamento do Dataset `train.csv`

Este projeto implementa um pipeline modular para carregamento, apresentacao, limpeza, tratamento de inconsistencias, deteccao de outliers e normalizacao do dataset `train.csv`.

## Estrutura do Projeto

```text
.
├── train.csv
├── main.py
├── data_loader.py
├── data_presenter.py
├── data_preprocessor.py
├── README.md
└── outputs/
    ├── cleaned_train.csv
    ├── invalid_values_report.csv
    ├── outliers_report.csv
    ├── normalization_report.csv
    ├── text_cleaning_report.csv
    └── preprocessing_summary.json
```

## Descricao dos Dados

O arquivo `train.csv` possui 92.106 registros e 70 colunas. A coluna `TARGET` representa a variavel alvo de classificacao binaria.

Distribuicao inicial do `TARGET`:

| Classe | Registros | Percentual aproximado |
| --- | ---: | ---: |
| 0.0 | 83.298 | 90,44% |
| 1.0 | 8.808 | 9,56% |

O dataset apresenta desbalanceamento de classes, com predominancia da classe `0.0`.

## Valores Invalidos

Registros antigos do banco de dados utilizavam codigos numericos para representar valores ausentes. Neste projeto, os seguintes valores sao tratados como dados invalidos e convertidos para `NaN`:

```python
[-99, -9998, -9999]
```

Embora a regra original mencione `-99` e `-9998`, a analise exploratoria identificou forte presenca de `-9999`, por isso ele tambem foi incluido como valor invalido.

## Regra de Remocao de Colunas

Colunas com mais de 80% de valores invalidos sao removidas do dataset final.

Regra:

```text
percentual_invalidos > 80% => remover coluna
```

Essa decisao evita manter variaveis com baixa cobertura informacional, reduz ruido e simplifica etapas posteriores de modelagem.

## Classes Implementadas

### `DataLoader`

Arquivo: `data_loader.py`

Responsavel por carregar o CSV e validar se o dataset possui linhas e colunas.

### `DataPresenter`

Arquivo: `data_presenter.py`

Responsavel por apresentar:

- quantidade de registros;
- quantidade de colunas;
- nomes das colunas;
- tipos de dados;
- registros nao nulos por coluna;
- uso de memoria;
- distribuicao da coluna `TARGET`.

### `DataPreprocessor`

Arquivo: `data_preprocessor.py`

Responsavel por:

- analisar valores invalidos;
- substituir valores invalidos por `NaN`;
- remover colunas com mais de 80% de invalidos;
- remover colunas identificadoras, como `HS_CPF`;
- limpar e padronizar textos;
- detectar outliers por IQR e Z-score;
- tratar outliers por clipping;
- imputar valores ausentes;
- normalizar colunas numericas;
- salvar relatorios e base final.

Todas as classes e metodos possuem docstrings no padrao Google, contendo entradas e saidas.

## Limpeza de Textos

As colunas textuais identificadas sao:

- `ORIENTACAO_SEXUAL`
- `RELIGIAO`

A limpeza textual aplica:

- remocao de espacos extras;
- conversao para maiusculas;
- remocao de acentos;
- padronizacao de multiplos espacos;
- preenchimento de valores ausentes com `DESCONHECIDO`.

Observacao: `ORIENTACAO_SEXUAL` e `RELIGIAO` sao variaveis sensiveis. A permanencia delas no dataset limpo nao significa que devam ser usadas automaticamente em modelos preditivos. Seu uso deve ser avaliado com criterios de etica, privacidade e fairness.

## Outliers

O projeto implementa dois metodos de deteccao:

### IQR

O metodo IQR utiliza o intervalo interquartil:

```text
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
```

Ele e recomendado como padrao para esta base porque varias colunas possuem distribuicoes assimetricas, muitos zeros e caudas longas. Exemplos:

- `ESTIMATIVARENDA`
- `DISTCENTROCIDADE`
- `DISTZONARISCO`
- `PIBMUNICIPIO`
- `MEDIARENDACEP`
- variaveis de renda familiar

### Z-score

O Z-score calcula quantos desvios padrao um valor esta distante da media:

```text
z = (x - media) / desvio_padrao
```

Valores com `|z| > 3` sao considerados outliers.

Neste projeto, o Z-score e mantido como metodo alternativo e comparativo. Ele e menos adequado como padrao porque assume distribuicoes mais proximas da normalidade, o que nao ocorre em muitas colunas da base.

## Tratamento de Outliers

O tratamento escolhido foi clipping, tambem chamado de winsorizacao:

```text
valores abaixo do limite inferior => limite inferior
valores acima do limite superior => limite superior
```

Esse metodo foi escolhido porque preserva linhas do dataset. Remover registros poderia descartar exemplos importantes, especialmente porque a classe `TARGET=1` e minoritaria.

Por seguranca, outliers sao avaliados apenas em colunas numericas com mais de 20 valores distintos. Assim, o processo evita tratar como outlier valores validos em colunas binarias ou discretas.

## Imputacao

Apos substituir os codigos invalidos por `NaN`:

- colunas numericas recebem imputacao pela mediana;
- colunas categoricas recebem imputacao pela moda;
- se nao houver moda disponivel, o valor `DESCONHECIDO` e utilizado.

A mediana foi escolhida por ser mais robusta em distribuicoes assimetricas.

## Normalizacao

O metodo padrao de normalizacao e `robust`, baseado em mediana e IQR:

```text
x_normalizado = (x - mediana) / IQR
```

Esse metodo e mais adequado para a base porque varias variaveis possuem caudas longas e outliers.

Tambem e possivel executar com normalizacao padrao:

```text
x_normalizado = (x - media) / desvio_padrao
```

Ou desativar a normalizacao.

## Como Executar

Carregar e validar os dados:

```bash
python3 main.py --load_data
```

Apresentar estrutura dos dados:

```bash
python3 main.py --data_present
```

Executar pre-processamento completo:

```bash
python3 main.py --pre-processing
```

Executar usando Z-score para tratamento de outliers:

```bash
python3 main.py --pre-processing --outlier_method zscore
```

Executar sem normalizacao:

```bash
python3 main.py --pre-processing --scaler_method none
```

Alterar o threshold de remocao de colunas:

```bash
python3 main.py --pre-processing --missing_threshold 0.80
```

Definir arquivo de entrada e diretorio de saida:

```bash
python3 main.py --data_path train.csv --output_dir outputs --pre-processing
```

## Arquivos Gerados

### `outputs/cleaned_train.csv`

Dataset final apos cleanup, tratamento de outliers, imputacao e normalizacao.

### `outputs/invalid_values_report.csv`

Relatorio com quantidade e percentual de valores invalidos por coluna.

### `outputs/outliers_report.csv`

Relatorio com deteccao de outliers por IQR e Z-score, incluindo limites, quantidades e percentuais.

### `outputs/normalization_report.csv`

Relatorio com os parametros de normalizacao aplicados em cada coluna numerica elegivel, incluindo metodo, centro, escala, estatisticas originais e estatisticas apos a normalizacao.

### `outputs/text_cleaning_report.csv`

Relatorio com alteracoes feitas em colunas textuais.

### `outputs/preprocessing_summary.json`

Resumo consolidado do pre-processamento, contendo:

- shape original;
- shape final;
- valores invalidos considerados;
- threshold utilizado;
- colunas removidas;
- colunas identificadoras removidas;
- metodo de normalizacao;
- colunas normalizadas;
- valores utilizados na imputacao.

## Decisoes Importantes

- `HS_CPF` e removido por ser identificador unico e nao deve ser usado como feature de modelagem.
- `TARGET` nao recebe normalizacao nem tratamento de outlier.
- IQR e o metodo recomendado para esta base.
- Z-score fica disponivel como comparacao e opcao de execucao.
- Colunas binarias e discretas de baixa cardinalidade nao passam por tratamento de outlier.
- O dataset contem variaveis sensiveis; seu uso em modelos deve ser avaliado antes da etapa de treinamento.
