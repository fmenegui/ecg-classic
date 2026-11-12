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
