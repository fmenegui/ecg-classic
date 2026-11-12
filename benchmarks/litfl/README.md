# Benchmark LITFL

102 ECGs reais da biblioteca do [Life in the Fast Lane](https://litfl.com/ecg-library/) (scans e fotos de livros, sites e aparelhos diversos), listados em `metadata.csv` (mesma lista do repositório `image-based-ecg-classification-emergency`). Não existe sinal verdadeiro, então o benchmark mede duas coisas:

**1. Estrutura, contra uma referência revisada à mão** (`reference.json`)

| Campo | O que é | Como foi anotado |
|---|---|---|
| `layout` | família impressa: `3x4`, `3x4+1`, `3x4+2`, `3x4+3`, `6x2`, `rows` | revisão visual da página inteira |
| `rhythm` | derivações das tiras de ritmo, de cima para baixo | rótulo impresso no início da tira, conferido em recorte; `null` quando ilegível |
| `speed` | mm/s | 25 em todas (texto impresso onde existe) |
| `gain` | ganhos distintos da página, mm/mV | altura do pulso de calibração e texto do cabeçalho; `null` quando o pulso está cortado |
| `duration` | segundos por linha | 10 (página de 12 derivações) |
| `review` | `bom` / `atencao` / `falha` | revisão visual de 12/11/2026, só como contexto |

A imagem 0043 está marcada como `indisponivel` (o site recusa o download) e fica de fora.

**2. Coerência interna, sem referência**

- **Cobertura:** fração das amostras que a página mostra e que saíram extraídas.
- **Einthoven:** |I − II + III| / RMS(I, II, III) nas medidas da grade antes da fusão. Num 3x4 as três derivações de membros estão na mesma coluna, no mesmo instante, então o resíduo mede erro de traçado, ganho e alinhamento.
- **Duplicata:** NRMSE entre a tira de ritmo II e a II da grade no mesmo trecho. Em várias montagens do LITFL a tira não é simultânea, por isso vale olhar a mediana e a variação entre versões, não o valor absoluto.

## Rodar

```bash
pip install -e .
python benchmarks/litfl/run.py --jobs 2                     # baixa as imagens para benchmarks/litfl/cache/
python benchmarks/litfl/run.py --compare benchmarks/litfl/baseline.json
python benchmarks/litfl/run.py --config nonrigid=poly       # qualquer campo do Config
```

Saída em `results/litfl/`: `results.json` (por imagem e resumo), `report.md` (tabela de métricas e imagens com erro) e `report.html` com `img/` (resumo + galeria das páginas retificadas com o traçado extraído; abre direto do disco). `--no-html` pula as imagens. O workflow `.github/workflows/litfl-benchmark.yml` roda o benchmark em todo PR que mexe em `ecgdig/` e publica o relatório no resumo do job, comparado com `baseline.json`.

## Limites

A referência foi feita por uma pessoa olhando a saída do próprio programa; erros sutis de rótulo podem ter passado. Velocidade e duração quase não discriminam (quase tudo é 25 mm/s e 10 s). As métricas de coerência detectam piora de traçado, mas não provam que o sinal está certo.
