# ecgdig — avaliação geral e roadmap de robustez

*Digitalizador clássico de ECG (sem deep learning). Avaliação de 12/11/2026.*

## 1. Estado atual

### O que funciona bem
- **Geometria pela grade:** fase local de Fourier + unwrapping + homografia (RANSAC) + polinômio não rígido. Retificação sub-milimétrica (resíduo 0,1–0,3 mm em fotos reais).
- **Linha central por inversão morfológica** (pena elíptica; abertura do topo / fechamento da base): preserva o pico do R.
- **Física usada explicitamente:** Einthoven/Goldberger, cópias duplicadas da mesma derivação, registro por batimentos simultâneos, fusão ponderada por confiança (oclusões).
- **Números:** 19–22 dB SNR no sintético (teste separado); em fotos reais, cópias independentes da mesma derivação concordam a 20–60 µV fora de oclusões (≈ limite de quantização de 1 px).

### Onde quebra (ordem de gravidade)
1. **Layout = lista fechada + heurística de contagem de linhas.** Não cobre Cabrera, 2 canais, 15 derivações, 6 tiras de ritmo, multipágina, impressão cortada, 50 mm/s, **ganho dividido** (precordiais a 5 mm/mV — amplitude sairia errada).
2. **10 s / 25 mm/s / 10 mm/mV fixos**; cabeçalho não é lido.
3. **Binarização perde informação** (grade vazando, tinta fraca, carimbo, manuscrito decididos por limiar).
4. **Rastreamento independente por linha**; cruzamentos/fusões tratados por remendos → é onde está o erro residual.
5. **Geometria:** unwrapping guloso pode deslizar períodos (cisalhamento); vinco, dobra e lente sem modelo.
6. **Origem temporal** depende da convenção do aparelho (±50 ms quando o traço sai colado ao pulso).
7. **Sem "não sei":** falta nota de qualidade e recusa.
8. **Validação fraca:** sintético mais fácil que o real; poucas fotos reais.

## 2. Modelos físicos: fazem sentido?

Sim, como **prior onde o pixel é ambíguo** (oclusões, fusões, fronteiras, picos cortados) e **teste de consistência global** — nunca sobrescrevendo pixels confiáveis (87–95% das amostras já estão no limite de quantização; um prior ali só pode apagar patologia: ST, onda delta, entalhes, espícula de marcapasso). O ganho esperado concentra-se nos 5–13% de amostras ocluídas, que dominam o erro residual.

| Modelo | Vale? | Por quê |
|---|---|---|
| Restrições lineares exatas (Einthoven/Goldberger, duplicatas) | Sim (feito) | Erro de modelo zero |
| **Template de batimento do próprio paciente** (mediana de ~10 batimentos, DTW restrito, clusters de morfologia) | **Maior ganho** | Específico do paciente → respeita patologia |
| **Dipolo / posto 3** (12 derivações ≈ L·d(t), L estimado por PCA robusta) | **Sim** | Resolve "qual traço é de qual derivação" nas fusões; valida layout (rótulo trocado quebra o posto 3) |
| Soma de gaussianas (McSharry 2003; versão 3D de Sameni) | Como representação do template | Como prior genérico enviesa arritmias (FA, ESV, marcapasso, bloqueios) |
| Tórax cilíndrico | Não | Uma imagem não tem dados para ajustar geometria de tórax; o L empírico já absorve isso; se faltar dado, Dower/Kors são priors melhores |
| EKF (Sameni et al. 2007) | Ideia certa, ferramenta errada | Offline pede suavizador (EKS/RTS); a observação (imagem) é multimodal → viola gaussianidade |

**Formulação proposta (MAP variacional):**
E(s) = Σ ρ(render(s) − imagem) + λ·w_baixa_conf·‖s − template∘warp‖² + λ₂·‖s − L·d‖², com Einthoven/Goldberger como restrições duras; ρ = Huber; IRLS ou L-BFGS. Alternativa leve: "Viterbi fisiológico" (transição prevista pelo template na fase do batimento).

## 3. Mudanças propostas

**A. Layout universal por hipótese e verificação** — layout como programa paramétrico (tiras: derivação, t₀, t₁, ganho, velocidade); hipóteses de separadores, OCR clássico dos rótulos, cabeçalho, forma dos pulsos (degrau = ganho dividido) e biblioteca de formatos (GE, Philips, Mortara, Schiller, Nihon Kohden, Cabrera); pontuação por rótulos + Einthoven + duplicatas + posto 3 + progressão precordial + alinhamento de batimentos; saída MAP com confiança, "derivação desconhecida" se fraca.

**B. Tinta × grade sem limiar** — desconvolução de "corantes" em densidade óptica (Macenko 2009, Vahadane 2016); RPCA (grade = baixo posto separável, traço/texto = esparso); mapas suaves até o fim; texto/carimbo por SWT/MSER.

**C. Rastreamento conjunto** — detector de Steger (centro, largura, direção sub-pixel); todas as linhas juntas (k caminhos disjuntos, Berclaz 2011, ou HMM acoplado com "toda tinta explicada"); fast marching para QRS íngremes.

**D. Geometria** — unwrapping com rotulação inteira global (PUMA, Bioucas-Dias 2007) + borda da grade (LSD); lente por plumb-line (Devernay & Faugeras 2001); papel curvo por superfície desenvolvível/cilíndrica; vincos por deslocamento suave por partes (TV); reflexo especular → dado faltante.

**E. Análise-por-síntese** — renderizador diferenciável (ecg-torch-renderer) otimizando sinal (B-splines), pena, PSF, warp e iluminação contra a foto, com os priors da seção 2.

**F. QC e incerteza** — σ por amostra (Laplace), flags por derivação, nota global, recusa.

**G. Validação** — sintético difícil (cruzamentos, ganho dividido, 50 mm/s, Cabrera, folha curva); PhysioNet 2024 (GT PTB-XL); pares reais impressos+fotografados; consistência não supervisionada; métricas clínicas (QRS, QT, ST, amplitudes) além do SNR.

## 4. Ordem de execução
1. A — layout universal *(primeira versão feita: biblioteca + Cabrera + 6 tiras + hipótese livre; OCR clássico; velocidade pelo texto; 24/24 no sintético misto; 6/7 imagens reais da internet — falha só sem rótulos impressos)*
2. B — mapas suaves
3. C + template de batimento + posto 3
4. D — geometria *(vincos feitos antes da hora, 12/11/2026: campo local por células de 5 mm, ideia do Undistortion da PMcardio reimplementada sobre a fase da grade — sem Dotter nem Gridder; sintético com dobras 9,7 → 17,3 dB, neutro em papel plano. Falta: PUMA, lente, reflexo)*
5. F — QC/incerteza (em paralelo)
6. E — inversão por renderização

## 5. Avaliação no PhysioNet (Kaggle), 13/11/2026

20 registros × 9 capturas do treino da competição, com sinal verdadeiro. A métrica é o SNR com alinhamento de ±0,2 s em passo de 1 amostra. Resultado: **3,3 → 11,0 dB** no geral; render 21,4; scans 10,6–12,6; fotos 3,4–10,8. O layout 3x4+1 passou de 44% para 92%. LITFL e sintéticos ficaram sem regressão.

A decomposição por oráculo (`benchmarks/physionet/diagnose.py`) apontou as causas, e cada uma foi corrigida:

- **Grade preta em scans P&B:** o molde agora desce até a linha.
- **Papel deslocado no scanner,** com a grade 0,3–1,2 mm fora do lugar, o que é ambíguo para a fase de 1 mm: foi criado o ajuste pelas linhas grossas (`_major_snap`).
- **Origem do tempo decidida por bordas falsas:** o pulso passou a ser âncora fraca.
- **Linhas espúrias em fotos:** borda impressa, cabeçalho e régua deixaram de ser tratados como linhas de ECG.
- **Duração por artefato:** a duração agora segue a duração padrão que os finais de linha confirmam.

O que ainda custa SNR, em ordem: distorção horizontal local (≈ 1 dB; ancorar colunas nas marcas separadoras), layout/derivação trocada (≈ 0,9 dB), fotos de celular (0005 ≈ 3,4 dB, com o próprio traço errado) e ganho (≈ 0,5 dB).

Lição de método: num render, 1 ms de deslocamento muda o SNR em vários dB. Comparações entre versões precisam de alinhamento por amostra e de sinais salvos para recalcular.

### Segunda rodada (13/11/2026): 11,2 → 13,2 dB

O ponto de partida foram os relatos do 4º e do 5º lugares do Kaggle (~22,7 dB, com redes neurais), cujo maior ganho veio da retificação local pela grade inteira. O equivalente clássico implementado aqui:

- **Grade:** cada linha de 1 mm localizada, com o traço mascarado por cor.
- **Fronteiras:** a marca separadora é reconhecida como retângulo.
- **Restrições físicas no Viterbi:** volta à linha de base e origem ancorada no pulso.
- **Fotos:** pulso e duração tratados de forma robusta.

Fotos de celular foram de 3,7 para 11,3 dB e scans coloridos de 12,1 para 15,2 dB.

O oráculo de geometria (scan registrado no render) mostra que o gargalo agora é o traçado: o scan fica em ~15 dB contra ~23 dB no render. Próximos passos:
1. estimador de traçado MMSE (forward-backward com a média a posteriori), testado na 3ª rodada;
2. máscara da folha pela presença de grade, para fotos com vinheta;
3. simultaneidade entre linhas onde a grade é fraca (P&B).

### Terceira rodada (13/11/2026): leitura do traço, +0,5 dB

Dois testes contra o erro de instante no QRS:

- **Média a posteriori do instante nos trechos íngremes** (centróide da tinta em cada linha de pixel, entre os pontos de coluna): 13,23 → 13,44 dB em 180 imagens. Fica opcional (`Config(steep_subpx=True)`), porque somado ao ajuste por síntese acrescenta só +0,03 dB.
- **Ajuste por síntese** (`ecgdig/synthesis.py`): modelo M = 1 − exp(−Σ a·g_σ) com pena e borrão estimados na imagem; os pontos da curva se deslocam pela normal até o desenho casar com a tinta. Resultados: 13,17 → 13,64 dB em 135 imagens locais e 14,07 → 14,53 dB em 45 imagens do Kaggle. Ligado por padrão, só em traço fino (raio ≤ 2 px). Sintéticos idênticos; LITFL com Einthoven 0,512 → 0,473.
- **Batimento médio como prior global:** piora de 0,1 a 0,9 dB e não entra.

Onde ainda há espaço: as fotos (0006, 0009, 0010) quase não ganham com a síntese. O limite delas é a geometria e a iluminação, não o traço.

### Quarta rodada (13/11/2026): rastreio e linhas, +0,9 dB

- **Diagnóstico do resto:** depois do oráculo, 40–48% do erro está nos picos, que ficam aquém do verdadeiro (nas fotos, 20–50% de perda nos 10% piores).
- **Translação rígida no Viterbi:** QRS gigante que cruza linhas não paga a regra de continuidade das bordas. Num render, 10,3 → 29,9 dB.
- **Ritmo comum entre linhas:** as linhas de um ECG multicanal são simultâneas, então uma linha da ponta sem QRS nas mesmas colunas e fora do espaçamento sai. Resolve fotos de tela e cabeçalhos sem depender do pulso.
- **Ponte por tinta contínua** e **segunda busca do pulso**, validada contra os outros pulsos.
- **Resultado em 135 imagens locais:** 13,64 → 14,57 dB. LITFL e sintéticos idênticos.
- **Piso da impressão:** nos scans, ±3 ms por batimento no instante do QRS, o mesmo em todas as capturas da mesma folha. A grade está certa; é o sinal impresso que saiu deslocado. Nenhum método recupera isso.
- **O que falta no clássico:** fotos com sombra/curvatura (resíduo de grade 0,35–0,5 mm por célula), ganho em fotos em perspectiva e o traço fraco de fotos de tela. O próximo salto de leitura do traço é o híbrido: um segmentador de traço aprendido sem template, com o resto clássico.

### Híbrido, primeiro teste (14/11/2026)

- **Montagem:** U-Net pequena (386 mil parâmetros) treinada só com recortes sintéticos genéricos (`hybrid/gen.py`: papel, grade, pena, texto, manchas, moiré e degradações sorteados; nenhuma imagem do PhysioNet/Kaggle). Ela entra no digitalizador como mapa de tinta (`Config(ink_model=..., ink_mix=...)`); o resto é clássico.
- **Treino:** CPU do GitHub (`hybrid-train.yml`), 5h20 → ~4.700 passos (4 s/passo). Ainda longe de convergir.
- **Resultado, 135 imagens locais, mistura 50% rede + 50% clássico:** 14,57 → 13,69 dB (67 melhores, 65 piores; mediana da diferença +0,05 dB).
  - Ganha onde o clássico falha: fotos de tela e amassadas (+8 a +10 dB em casos isolados), scan colorido +1,0 dB.
  - Perde no que já era bom: render −2,8 dB, foto de celular −3,1 dB, com algumas quebras estruturais (−15 dB).
- **Melhor dos dois por imagem (oráculo):** 15,49 dB. Há ~0,9 dB a ganhar com uma fusão que decida sem gabarito.
- **Próximos passos:** (1) usar a rede só na leitura do traço, mantendo o mapa clássico para linhas, layout, pulsos e tempo; (2) escolher ou fundir por imagem com critérios internos (Einthoven, II duplicado, ritmo comum entre linhas); (3) treino em GPU (dezenas de milhares de passos).

## 6. PM-ECG-ID (PMcardio), 14–15/11/2026

**A base:** Zenodo 13617673. São 6.600 imagens de 100 ECGs do PTB-XL em 60 cenários (fotos, JPEG, blur, brilho, contraste, padding, perspectiva, rotação, escala e variantes digitais).

**A métrica:** a mesma do artigo da PMcardio. O alinhamento é de ±0,5 s, com a mediana removida, e cada derivação que falha conta como 0. O script é `benchmarks/pmcardio/compare.py`.

**A referência publicada** (Demolder et al., medRxiv 2024.08.31.24312876) é PCC > 0,91, SNR > 12,5 dB e falha em 6,62% das derivações.

### Amostra de 120 ECGs (2 por cenário)

| versão | SNR | PCC | falha | SNR sem falhas |
|---|---|---|---|---|
| dev4 (início) | 4,60 | — | — | — |
| dev11 | 5,82 | 0,567 | 20,1% | 7,28 |
| dev16 | 6,99 | 0,632 | 15,7% | 8,29 |
| dev18 | 7,14 | 0,635 | 16,1% | 8,51 |
| dev20 | **7,50** | **0,653** | **13,1%** | **8,63** |
| PMcardio (publicado, base inteira) | > 12,5 | > 0,91 | 6,6% | — |

Ainda há ~5 dB de distância. O PhysioNet segue em 14,57 dB no dev20. Os números do artigo por grupo parecem calculados só sobre as digitalizações bem-sucedidas.

### Base inteira (6.600 imagens, GitHub Actions, commit 7a60059)

| | SNR | PCC | falha | SNR sem falhas |
|---|---|---|---|---|
| ecgdig (clássico) | 6,99 | 0,591 | 22,2% | 8,99 |
| PMcardio (publicado) | > 12,5 | > 0,91 | 6,6% | — |

Tabela completa por família e cenário: `docs/resultados/pm-ecg-id-7a60059.md`.

- A falha na base inteira (22%) é bem maior que na amostra de 120 (12,6%): 807 ECGs falham inteiros, 487 deles com layout "rows" (nº de linhas detectadas não bate com nenhum formato). Só 24 por tempo esgotado.
- Por formato, o 6x2 de uma página (31 dos 100 ECGs, 6,0 dB) responde por ~2 dB dos 5,5 dB que faltam; 3x4+1R (29 ECGs, 9,1 dB) ~1 dB; 3x4 de uma página (5,4 dB) ~0,9 dB; 12x1 (3,9 dB) ~0,8 dB.
- ECGs com deriva de linha de base grande no próprio sinal (> 1,4 mV em 2,5 s) ficam ~5 dB abaixo: os traços sobem inclinados em relação à grade e as linhas vizinhas se cruzam; a detecção de linhas por projeção horizontal supõe linhas retas.

### Correções clássicas desta rodada

- **Particularidades do gerador PM:**
  - vão comum entre colunas retirado da imagem;
  - coluna de pulsos solta do traço apagada; ela dá o ganho;
  - âncora no fim da linha para páginas de 10 s;
  - rampa de extrapolação limitada.
- **Layout:**
  - evidência de fronteira relativa ao fundo da linha e pesada pela coerência entre linhas;
  - física como log-verossimilhança com piso;
  - Cabrera só com o rótulo −aVR;
  - onda T depois do QRS como desempate de 0/180°.
- **Escala resgatada:** a estimativa global às vezes toma o passo de 5 mm por 1 mm. O resgate ×5 acertava a escala, mas desligava homografia e refinamento. Agora as correspondências usam a fase de 1 mm ou de 5 mm na escala corrigida. Grade preta e pontilhada passaram de −3 para +9 dB e de −0,7 para +22 dB.
- **Retificação não-rígida:** o polinômio cúbico extrapolava longe dos blocos medidos e deformava a folha. Agora ele fica no intervalo observado, mas só quando a escala local sai de 0,7–1,4. Em fotos com perspectiva forte a extrapolação é necessária. Uma página foi de −0,8 para 17 dB.
- **Folha pequena sobre papelão:** quando o GrabCut vaza, entra Otsu na cor do fundo. A foto é recortada à folha quando ela ocupa menos de 30% do quadro.
- **Traço claro:** o piso do Otsu passou a ser relativo ao contraste do traço. Com poucas linhas longas, testa-se um limiar menor, que fica se revelar mais linhas em passo regular.
- **Pulsos só em linhas alternadas:** as linhas no meio do passo entram, desde que o passo seja ≥ 40 mm.
- **Deriva de linha de base:** em fotos amassadas, o rastreio que salta de linha ou segue um vinco deixa um desvio lento de 1–3 mV. Onde a mediana móvel de 0,6 s anda mais de 1,5 mV, ela é subtraída. A deriva é medida longe das bordas do trecho; nas bordas a mediana segue o QRS e disparava em ECGs limpos do PhysioNet (−0,2 dB). Fotos: 4,7 → 5,9 dB.
- **Blur:** quando não sobra linha, o nível da grade é recalculado sem a sombra do traço borrado. Ganhos raros (2,5 e 40 mm/mV) só valem pelo pulso impresso à esquerda; o fim do traço borrado virava um "pulso" de 2,5 mm/mV (uma página −6,7 → +4,3 dB).
- **Avaliador:** `max_tasks_per_child` travava o executor do Python 3.11 após 12 tarefas. Agora um executor novo é criado por lote, com prazo por imagem.

### O que testar sempre

Cada mudança mexe em várias páginas para os dois lados; uma página isolada muda vários dB com 1% de redimensionamento. A decisão é pelo agregado da amostra (~1 h local, 2 núcleos). Cada mudança passa também pelo PhysioNet (135 imagens, 14,57 dB) e pelo LITFL.

Exemplo: relaxar a limpeza de linhas de grade ajudou no PM, mas apagou linhas de base no PhysioNet (11,1 → 3,4 dB) e foi revertido.

### O que ainda falha (amostra)

| família | falha |
|---|---|
| blur 31/41 | ~50% |
| perspectiva 30/40 | ~35% |
| fotos de grade preta pontilhada | restos da grade 5 mm viram linhas falsas |
| 6x2 com colunas em alturas próprias ou derivações sobrepostas | — |
| opacidade digital 0,15 | traço quase invisível |

## 7. Velocidade, determinismo e nota de qualidade, 15/11/2026

Branch `claude/classico-melhorias`, a partir do `r4` (f833818). Nenhuma mudança de precisão nesta rodada: o objetivo foi fazer o mesmo cálculo em metade do tempo e medir onde o erro do PM-ECG-ID está.

### Velocidade: 1,9× no total, com a mesma saída

Oito páginas, um processo, máquina de 2 núcleos sem outra carga:

| página | r4 | agora |
|---|---|---|
| sintético 3x4+1 (3 páginas) | 12,6–14,1 s | 7,0–7,3 s |
| PM-ECG-ID, foto Samsung | 40,2 s | 16,8 s |
| PM-ECG-ID, scan | 43,3 s | 17,3 s |
| PM-ECG-ID, foto iPhone | 46,2 s | 22,5 s |
| PM-ECG-ID, rotação 5° | 13,6 s | 8,3 s |
| PM-ECG-ID, digital de 7483 px | 99,0 s | 62,7 s |
| **total** | **282,9 s** | **149,1 s** |

O que mudou, sempre com o resultado conferido contra a versão anterior:

- **`_major_snap`:** `np.nanmedian` por faixa virou uma ordenação (`_nanmedian_axis`); o laço por linha da grade foi vetorizado; `generic_filter(np.nanmedian)` virou janelas 3×3. `rectify` 6,6 → 3,4 s.
- **`ink_map`:** o `median_filter` 4D (5,5,1,1) virou `cv2.medianBlur` 5×5 por fase do molde. 3,6 → 0,8 s.
- **`column_runs`:** todas as colunas de uma vez (`diff` + `reduceat`).
- **`viterbi_track`:** contas em escalares do Python em vez de escalares do NumPy.
- **`synth_refine`:** gaussiana 2D como produto de duas 1D e blocos calculados uma vez por iteração. 3,3× mais rápida com a mesma conta (diferença máxima de 6e-13 px).

Em scans e fotos reais a síntese era 76% do tempo (26,8 de 35,3 s num scan do PM). Continua sendo a maior parcela.

### A síntese dependia do relógio

`synth_refine` parava depois de 4 s de relógio por segmento, então o número de iterações, e com ele o sinal, dependia da velocidade e da carga da máquina. O corte acontecia de fato: num scan do PM os 13 segmentos levavam 32 s sem limite e 26 s com ele. Agora o limite é de trabalho (`SYNTH_WORK`, pares ponto-pixel × iterações), igual em qualquer máquina; o relógio fica como freio de emergência.

### Equivalência medida

| conjunto | r4 | agora |
|---|---|---|
| Sintético, 30 páginas (medium, heavy, fold, mix), SNR | 20,796 dB | 20,797 dB |
| Sintético, layouts certos | 30/30 | 30/30 |
| PM-ECG-ID, 60 ECGs (1 por cenário): SNR / PCC / falha | 8,12 dB / 0,695 / 8,3% | 8,12 dB / 0,695 / 8,3% |
| PM-ECG-ID, layouts iguais ao r4 | — | 60/60 |
| PM-ECG-ID, maior diferença por ECG | — | 0,13 dB |

Com `Config(synth=False)` a saída é idêntica bit a bit à do r4 (conferido em 3 páginas). Com a síntese ligada as diferenças vêm do limite determinístico.

### Onde está o erro do PM-ECG-ID: no layout

Os mesmos 60 ECGs, separados por o layout lido bater ou não com o formato da base:

| | ECGs | SNR | PCC | falha |
|---|---|---|---|---|
| layout certo | 41 | 11,77 dB | 0,864 | 0,0% |
| layout errado | 19 | 0,26 dB | 0,330 | 26,3% |

Com o layout certo o clássico faz 11,8 dB e PCC 0,86; a referência publicada pela PMcardio para a base inteira é > 12,5 dB e PCC > 0,91. Um terço das páginas perde tudo no layout. A rodada da base inteira no CI (run 43742643092, `r4` f833818, 5.994 ECGs) confirma a proporção:

| base inteira | ECGs | SNR | PCC | falha |
|---|---|---|---|---|
| todos | 5.994 | 6,99 dB | 0,591 | 22,2% |
| 1 página, layout certo | 3.424 (57%) | 11,90 dB | 0,867 | 0,3% |
| 1 página, layout errado ou não lida | 2.212 (37%) | −0,37 dB | 0,181 | 55,1% |
| várias páginas | 358 (6%) | 5,53 dB | 0,490 | 27,7% |

A amostra de 60 ECGs desta seção (8,12 dB) é mais fácil que a base (6,99 dB). Em 4,0% dos ECGs alguma página termina em erro: 134 "nenhuma linha de traçado", 60 "nenhum segmento", 27 por tempo, 23 por escala implausível. Os erros mais comuns: 6x2 lido como 6 tiras inteiras (`6x1-prec`, `6x1-limb`: 6 derivações somem), tira de ritmo inventada (`3x4+1`, `3x4+2`, `6x2+1`, `6x2+2`) e 6x2 lido como 3x4+k.

Três causas vistas nas imagens de depuração:

- **Máscara da folha cortando a grade.** Em 6 das 131 páginas da amostra a máscara fica com um quadrilátero torto que deixa de fora os pulsos e o começo das linhas (`JPEG_45/img_65`, `JPEG_2/img_8`, `blur_21/img_81`, `brightness_120/img_13`, `gamma_0.5/img_69`, `doogee/img_24`). Sem pulsos e sem rótulos, o layout é decidido "só pela fisiologia".
- **Linha de base que anda** (`low_freq_noise_large`): a projeção horizontal acha 7 linhas em vez de 6, e só 4 dos 6 pulsos são encontrados.
- **Imagem estourada** (`brightness_160`): a metade esquerda do traço quase some e as linhas começam no meio da página.

Uma guarda que estendia a máscara até a área quadriculada (cor do papel + densidade de linhas) corrigiu a máscara nessas páginas, mas no A/B deu 8,12 → 7,99 dB (2 ECGs melhores, 2 piores): as páginas corrigidas continuam falhando mais adiante. Foi retirada.

### Nota de qualidade (`res["quality"]`)

`ecgdig/quality.py` dá uma nota de 0 a 1 por derivação e por página, sem gabarito. O sinal não muda; `Config(refuse_below=...)` devolve NaN abaixo da nota e fica desligado.

Medida contra o gabarito nos 60 ECGs (660 derivações lidas, "ruim" = PCC < 0,5, 22,6% delas):

| parte | AUC |
|---|---|
| confiança do layout | 0,76 |
| resíduo de Einthoven antes da fusão (só onde existe, 48%) | 0,74 |
| repetição dos batimentos | 0,72 |
| energia nos QRS da página | 0,59 |
| degraus | 0,55 |
| nota final (cobertura × repetição × confiança) | 0,75 |

Ainda é pouco para recusar com segurança: recusar os 20% piores sobe o SNR das aceitas de 8,7 para 9,8 dB e deixa passar 18% de derivações ruins. O motivo é o mesmo da seção anterior: no layout trocado cada traço é um ECG bem formado com o nome errado, e nenhuma medida feita derivação a derivação enxerga isso.

### LITFL

Nesta máquina o site liberou 89 das 102 imagens. Nelas, r4 e esta branch dão a mesma estrutura em 89/89 (layout, linhas, tiras de ritmo, velocidade, duração e ganhos), o mesmo resíduo de Einthoven em todas e a mesma mediana de ritmo × grade (0,519). O tempo por imagem foi de 17,8 para 7,2 s.

| LITFL, 89 imagens | r4 | agora |
|---|---|---|
| Layout / ritmo / velocidade / duração / ganho | 92,1 / 93,0 / 98,9 / 98,9 / 96,6% | iguais |
| Layout + velocidade + duração + ganho | 91,0% | 91,0% |
| Einthoven antes da fusão (mediana) | 0,474 | 0,474 |
| Segundos por imagem | 17,8 | 7,2 |

### Testado e descartado

- **Reamostragem única a partir da imagem original** (o mapa não-rígido composto com a homografia, uma interpolação cúbica em vez de duas): 20,797 → 20,806 dB nos 30 sintéticos (fold +0,25, medium −0,16). Neutro; não entrou.
- **Máscara estendida até a grade:** −0,13 dB no PM-ECG-ID (acima).

### O que não foi medido

O PhysioNet (Kaggle) não roda fora do CI, porque precisa do token. As mudanças desta rodada não alteram o sinal além do limite da síntese, mas a rodada de 135 imagens deve ser repetida no workflow antes do merge.

### Ferramenta nova

`benchmarks/pmcardio/ab.py <rodada A> <rodada B>` compara duas rodadas do `eval_pm.py` ECG a ECG (resumo, quantos melhoraram ou pioraram, layout lido em cada uma).


## 8. Estrutura da página: os erros de layout do PM-ECG-ID, 15/11/2026

A seção 7 mostrou que o erro do PM-ECG-ID está no layout: com o layout certo o clássico faz 11,9 dB; com o layout errado, −0,4 dB, e isso acontece em 37% dos ECGs. Esta rodada ataca essa parte. O conjunto de trabalho foi o cenário mais limpo da base, `photos_scans` (100 ECGs, 110 páginas, scanner de mesa): se a página é lida errada ali, o motivo é a estrutura, não o ruído. Das 94 páginas únicas, 21 saíam com o layout errado.

### Resultado

| conjunto | antes | depois |
|---|---|---|
| PM-ECG-ID `photos_scans`, 100 ECGs: SNR | 11,72 dB | **14,06 dB** |
| PCC | 0,770 | **0,875** |
| derivações em falha | 11,4% | **5,1%** |
| páginas únicas com layout certo | 73 de 94 | **87 de 94** |
| ECGs com ≥ 12,5 dB | 61 | 72 |
| PM-ECG-ID, 60 ECGs (um por cenário): SNR / PCC / falha | 8,12 dB / 0,695 / 8,3% | 8,39 dB / 0,713 / 8,8% |
| LITFL, 89 imagens reais: layout / tiras de ritmo | 0,921 / 0,930 | 0,933 / 0,965 |
| sintéticos, 30 páginas | 20,797 dB | 20,796 dB |

"Antes" é o commit 095f96c (seção 7) com o `find_words` novo. Nos scans, 17 ECGs melhoraram mais de 1 dB e 1 piorou (0,4 → −0,9 dB, uma página que já não era lida). No LITFL nenhuma imagem piorou; 3 passaram a ter layout ou tira de ritmo certos. A referência publicada pela PMcardio (> 12,5 dB, PCC > 0,91, 6,6% de falha) é da base inteira, com os 60 cenários; o número dos scans não se compara com ela. A base inteira e o PhysioNet só rodam no CI.

### O que estava errado e o que mudou

**1. Retificação: a semente do desembrulho da fase caía numa ilha** (`geometry._unwrap_quality`). O desembrulho 2D partia do melhor bloco da página. Em 5 dos 110 scans esse bloco fica numa região pequena e separada (margem, rodapé), o preenchimento cobria 10 a 30 blocos de 600 e a homografia saía de uma dúzia de pontos amontoados: página cisalhada, "escala implausível" ou 0 correspondências. Agora a semente é o melhor bloco da maior região conexa de blocos bons; se ela ainda deixa de fora mais de 30% dos blocos, as outras regiões entram com o inteiro escolhido pelo plano já desembrulhado (só quando esse plano ajusta com resíduo < T/8, ou seja, fora de papel amassado). Scans 8, 10, 20, 47: de −0,1 / 0,0 / −4,7 / 0,0 dB para 10,1 / 20,2 / 25,7 / 11,1 dB. Fotos amassadas, dobradas e com perspectiva conferidas: saída idêntica.

**2. Colunas com alturas próprias** (`colalign.col_struct`). Em parte das páginas do PM-ECG-ID cada coluna tem as suas linhas de base (6x2 com a coluna da esquerda apertada e a da direita espaçada; 3x4 com as quatro colunas desencontradas). A projeção horizontal mistura tudo e o layout sai `3x4+2`, `6x1`, `3x4+3`. O `col_align` que já existia depende de vãos em branco entre as colunas, que nessas páginas não existem (grade escura, rótulo e QRS no vão). O `col_struct` varre a página em janelas de 20 mm, encadeia os picos de cobertura em trilhos (altura, início, fim) e toma como fronteira de coluna o ponto em que vários trilhos acabam e outros começam. As colunas são realinhadas para alturas de referência com o maior intervalo entre as colunas (sem sobrepor faixas, aumentando a altura da imagem se preciso), e as linhas e o número de colunas seguem como conhecidos para a detecção de linhas (`_rows_from_hint`) e para o motor de layout (`col_hint`). Scans 23, 49 (6x2) e 32, 50, 77 (3x4): de −1,9 / 4,8 / −2,2 / 0,0 / −1,7 dB para 9,1 / 19,3 / 17,3 / 23,9 / 16,0 dB.

**3. Pés de pulso errados trocavam linhas certas** (`PULSE_ROWS_GUARD`). Com a grade preta comendo os pulsos, o `pulse_rows` achava 4 "pés" em alturas quaisquer e substituía as 12 linhas regulares da projeção por 3. Agora, quando a projeção dá 4 ou mais linhas em passo constante (12%) e os pés não caem nelas ou são menos da metade, vale a projeção.

**4. Página em pé girada 180° por engano** (`Config.orient_check`). Sem pulso nem rótulo que desempate, a orientação era decidida pela onda T, que erra. Quando pulsos e rótulos diferem em no máximo 2 entre as duas orientações, a outra orientação é lida até o layout (a retificação e a binarização são reaproveitadas) e a primeira decisão só é trocada se a outra for claramente melhor (margem de 5 na pontuação); para girar exige-se ainda que a física feche na orientação girada (≥ 0,5) e não na original. Scan 33: −2,8 → 11,2 dB. Oito scans girados 180° de propósito continuam sendo desvirados.

**5. Moldura lida como linha** (`layout_engine.ROW_DROP`). O motor de layout ganhou hipóteses que deixam de fora a primeira ou a última linha. Só valem quando todas as outras linhas têm pulso de calibração e a descartada não, ela está mais perto da vizinha que o passo, o layout que sobra é de 12 derivações e o fechamento de Einthoven/Goldberger melhora 0,25. Scan 66: −1,6 → 5,8 dB.

**6. `ocr.find_words`** agrupava componentes com uma varredura quadrática (41 de 89 s numa página digital com muito texto). Com as caixas ordenadas e a busca limitada por bisseção: 3,39 s → 0,12 s, mesmas palavras (teste contra a varredura original).

### Guardas que a validação exigiu

Cada mecanismo novo errou em alguma página boa antes de ficar como está. O que fechou cada caso:

- `col_struct` numa página comum (`brightness_-80/img_41`, scans 11 e 60): a linha é interrompida pelo pulso e pelo rótulo na troca de coluna e vira dois trilhos. Agora a estrutura é recusada se metade ou mais dos trilhos recomeça na mesma altura (± 2 mm) do outro lado, se as colunas não têm a mesma largura (± 20%), se a fronteira mais forte tem menos eventos que o dobro das linhas que atravessam a página, ou se as linhas de sempre já ficam a menos de 40% do passo de todas as colunas (a extração dá conta sem realinhar).
- Descartar linha tirava a tira de ritmo de páginas reais (LITFL 0023, 0026, 0027, 0042) e fechava páginas de 4 linhas como `3x1` (scans 17, 81). Daí a exigência do pulso como testemunha e de layout de 12 derivações.
- A checagem de orientação girava uma página de três precordiais (lidas de cabeça para baixo imitam I, II, III com fechamento 0,35) e desfazia um giro certo quando o Cabrera fechava a física na leitura invertida. Daí o piso de 0,5 e a regra de só trocar a primeira decisão com margem.

### Custo

Scans: mediana de 23 → 17 s por página na mesma máquina (2 processos em 2 núcleos; a base foi medida com concorrência, então a diferença real é menor). LITFL: 7,2 → 9,4 s por imagem, porque muitas páginas reais não têm pulso nem rótulo legível e passam pela segunda leitura até o layout.

### O que ainda falha nos scans (14 ECGs abaixo de 5 dB)

- 76 (6x2 com alturas próprias e 4 linhas que atravessam), 24 (3x4): o `col_struct` recusa a estrutura.
- 59, 79 (6x2 lido como 3 tiras), 70, 45, 6 (12x1 e 6x2 com linhas a 5-8 mm): detecção de linhas.
- 27 (3x4 com 2 linhas a mais), 13 (6x2 com pulsos nas duas colunas), 44 (layout certo, PCC 0,83, escala errada), 89 (2 páginas).
- 35, 97, 30: layout certo com 4 a 5 dB; é extração, não estrutura.

### O que não foi medido

A base inteira do PM-ECG-ID (5.994 ECGs) e o PhysioNet (Kaggle) só rodam no CI. A amostra de 60 ECGs subiu pouco (8,12 → 8,39 dB): fora dos scans, o layout também falha por máscara da folha, linha de base que anda e imagem estourada (seção 7), que esta rodada não tocou.
