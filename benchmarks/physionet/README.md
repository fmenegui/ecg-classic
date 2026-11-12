# Benchmark PhysioNet (Kaggle)

Treino da competição *PhysioNet - Digitization of ECG Images* (Kaggle): cada registro tem o sinal verdadeiro (`<id>.csv`, 12 derivações, 250–1025 Hz, 10 s; vazio onde o papel não mostra a derivação) e 9 imagens do mesmo ECG em tipos de captura diferentes (`<id>-NNNN.png`). Todas as páginas são 3x4 + ritmo II.

Roda no GitHub Actions (`.github/workflows/physionet-kaggle.yml`, manual) com o secret `KAGGLE_API_TOKEN` e a variável `KAGGLE_USERNAME` do repositório:

- `mode=inspect`: lista os arquivos e baixa um registro de exemplo;
- `mode=eval`, `n=20`: sorteia N registros (semente 2026), digitaliza as 9 imagens de cada um e mede SNR; `config=layout=3x4+1` força o layout.

**Métrica**: SNR por imagem = 10·log10(Σ sinal² / Σ ruído²) somando as 12 derivações, só nas amostras mostradas. "Alinhado": por derivação, deslocamento de até ±0,2 s e diferença de média removida (a métrica da competição tolera o mesmo). Reimplementação: o placar oficial pode diferir em detalhes. Referência externa: uma solução com deep learning em 24º lugar de 1.425 times relata 19,7 dB no placar privado.

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
