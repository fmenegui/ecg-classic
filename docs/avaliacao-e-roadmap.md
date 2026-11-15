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
| dev16 | **6,99** | **0,632** | **15,7%** | **8,29** |
| PMcardio (publicado, base inteira) | > 12,5 | > 0,91 | 6,6% | — |

Ainda há ~5,5 dB de distância. Os números do artigo por grupo parecem calculados só sobre as digitalizações bem-sucedidas.

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
