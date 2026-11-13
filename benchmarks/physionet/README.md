# Benchmark PhysioNet (Kaggle)

Treino da competição *PhysioNet - Digitization of ECG Images* (Kaggle): cada registro tem o sinal verdadeiro (`<id>.csv`, 12 derivações, 250–1025 Hz, 10 s; vazio onde o papel não mostra a derivação) e 9 imagens do mesmo ECG em tipos de captura diferentes (`<id>-NNNN.png`). Todas as páginas são 3x4 + ritmo II.

Roda no GitHub Actions (`.github/workflows/physionet-kaggle.yml`, manual) com o secret `KAGGLE_API_TOKEN` e a variável `KAGGLE_USERNAME` do repositório:

- `mode=inspect`: lista os arquivos e baixa um registro de exemplo;
- `mode=eval`, `n=20`: sorteia N registros (semente 2026), digitaliza as 9 imagens de cada um e mede SNR; `config=layout=3x4+1` força o layout.

**Métrica**: SNR por imagem = 10·log10(Σ sinal² / Σ ruído²) somando as 12 derivações, só nas amostras mostradas. "Alinhado": por derivação, deslocamento de até ±0,2 s (passo de 1 amostra) e diferença de média removida (a métrica da competição tolera o mesmo). Com passo grosso (4 ms, usado até a rodada de 12/11), um render variava ~6 dB com 1 ms de deslocamento. Reimplementação: o placar oficial pode diferir em detalhes. Referência externa: uma solução com deep learning em 24º lugar de 1.425 times relata 19,7 dB no placar privado.

## Diagnóstico e relatório

O artefato de cada rodada traz `results.json` (diagnóstico por imagem: layout, ganhos, duração, ppmm, resíduos da grade, origens por linha), `sig/*.npz` (verdade e previsão a 250 Hz), `img/*.jpg` (sobreposições) e `raw/` (imagens brutas e gabarito dos 5 primeiros registros, para depurar localmente).

```bash
gh run download <run_id> -R fmenegui/ecg-digitalization-classic -D art
python benchmarks/physionet/diagnose.py art/physionet-eval-<run_id>        # -> decomposition.json
python benchmarks/physionet/report_html.py art/physionet-eval-<run_id> \
    --before <artefato_anterior> --findings benchmarks/physionet/diagnostico.html -o relatorio.html
```

`diagnose.py` mede quanto SNR custa cada tipo de erro: aplica correções "oráculo" cumulativas (cada uma usa o gabarito só para o seu erro) na ordem layout/derivação trocada → trechos vazios → origem do tempo → escala horizontal → distorção horizontal local (janelas de 0,5 s) → ganho → linha de base. O que sobra é erro de forma.

## Rodada atual (13/11/2026, mesmos 20 registros)

| Tipo de imagem | Antes | Depois |
|---|---|---|
| 0001 (render) | 21,4 dB | 21,4 dB |
| 0003 (scan colorido) | 6,7 | 11,8 |
| 0004 (scan P&B) | −2,5 | 10,7 |
| 0011 (scan com sujeira) | 5,5 | 12,6 |
| 0012 (scan escurecido) | −4,4 | 10,6 |
| 0005 (foto de celular) | −3,9 | 3,4 |
| 0006 (foto de tela) | 5,8 | 8,8 |
| 0009 (foto amassada) | 0,7 | 9,0 |
| 0010 (foto com sombra) | 0,4 | 10,8 |
| **Todas** | **3,3 dB** | **11,0 dB** (11,2 na resolução original) |

Antes e depois recalculados com a mesma métrica sobre os sinais a 250 Hz. Layout 3x4+1 reconhecido: 44% → 92%. LITFL e sintéticos sem regressão (LITFL idêntico; heavy 19,4 → 20,3, medium 21,6 → 22,1, fold e mix iguais).

Causas encontradas e correções (detalhes em `diagnostico.html`):

1. **Grade preta** (scans P&B/escurecidos): o molde da grade parava em 30% de transmitância e a linha grossa virava tinta. Agora o molde desce até a linha quando a grade é escura, alargado ±1 px, e as bordas do traço usam só a tinta binarizada.
2. **Papel deslocado no scanner**: grade fora do lugar em 0,3–1,2 mm numa faixa da folha, ambíguo para a fase de 1 mm. Nova etapa `_major_snap` em `geometry.py` localiza cada linha grossa (5 mm, sem ambiguidade) em faixas de 20 mm. Só atua com desvio de 0,15–0,8 mm e ≥ 6 px/mm.
3. **Origem do tempo**: com o traço colado ao pulso, só os finais de linha decidiam. O fim do pulso entra como âncora fraca: escolhe o grupo na votação e entra na duração.
4. **Linhas espúrias nas fotos**: borda da área impressa, cabeçalho isolado e régua de texto colada. Saem quando estão na ponta do bloco, sem pulso, enquanto as outras linhas têm.
5. **Duração**: quando a mediana dos finais de linha cai num artefato, vale a duração padrão que mais finais confirmam.

## Primeira rodada (12/11/2026, 20 registros, 180 imagens)

| Tipo de imagem | Layout automático | Layout forçado 3x4+1 |
|---|---|---|
| 0001 (render limpo) | **21,3 dB** | 20,6 dB |
| 0003 (scan) | 6,7 | 6,6 |
| 0004 (scan cinza, ruidoso) | −3,0 | −2,0 |
| 0005 (foto de celular) | −4,2 | −6,9 |
| 0006 (foto de tela) | 5,8 | 6,6 |
| 0009 (foto) | 0,7 | 2,1 |
| 0010 (foto, papel manchado) | 0,3 | −2,6 |
| 0011 (scan, manchas) | 5,4 | 5,4 |
| 0012 (scan cinza, manchas) | −6,6 | −2,9 |
| **Todas** | **3,2 dB** | **3,0 dB** |

Forçar o layout certo não muda a média: o gargalo nas imagens reais é o sinal, não o layout. Diagnóstico nos 4 primeiros registros (sinais salvos em `sig/` no artefato):

1. **Origem do tempo errada em fotos** (ex.: 994226251-0006 deslocado ~0,2 s): fora da janela de alinhamento, a derivação inteira vira ruído.
2. **Deriva horizontal local em scans** (ex.: 426433979-0003, V5: alinhado em 8,2 s, −30 ms em 8,9 s, −45 ms em 9,6 s): escala horizontal não uniforme perto da borda, que o polinômio global não pega e o campo por células (com trava de 0,06 mm) não chega a aplicar.
3. **Amplitude subestimada** nos scans (ajuste predito→verdade 0,8–0,95) e picos de R cortados nas fotos.
4. **Layout errado em fotos e scans cinza** (6x1, "rows", 3x4+2) quando o layout é automático.
