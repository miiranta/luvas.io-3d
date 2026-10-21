# Scan 3D — reconstrução 3D no celular (extra do trabalho de Visão Computacional)

App Angular que grava um "vídeo" andando em volta de um objeto ou de um ambiente e devolve, no
próprio celular, um modelo 3D completo: nuvem esparsa com as poses das câmeras, nuvem densa e
malha texturizada, com métricas e exportação GLB/PLY/JSON.

O celular entra com o que ele tem de melhor — a **pose** de cada quadro (odometria
visual-inercial do ARCore, em metros e alinhada com a gravidade) e, quando existe, um mapa de
profundidade grosseiro — e a visão computacional faz o resto: características, emparelhamento
guiado pela geometria epipolar, triangulação, bundle adjustment, estéreo multivista na GPU,
fusão volumétrica, malha e textura. Nada vai para servidor: CPU num Web Worker, GPU por WebGPU.

## Como rodar

```bash
npm install --legacy-peer-deps      # o resolvedor de peers do npm 10 quebra na árvore do vitest
npm start                           # http://localhost:4200 (só a interface; sem AR no desktop)
npm run start:https                 # HTTPS em 0.0.0.0 — necessário para abrir pelo celular
npm run build                       # build de produção em dist/v1/browser
```

**Requisitos no celular:** Chrome no Android com ARCore (Google Play Services for AR) e WebGPU
(Android 12+, GPU Qualcomm/ARM). WebXR exige contexto seguro: pelo celular use
`npm run start:https` e aceite o certificado, ou o deploy atrás do proxy HTTPS. O iPhone não tem
WebXR AR no Safari, então o app avisa e não abre a captura.

```bash
npm test -- --watch=false           # 20 testes unitários (Vitest)
npm run typecheck                   # app, worker, testes e ferramentas
npm run verify                      # pipeline inteiro numa cena sintética, Chrome headless + WebGPU
npm run e2e                         # o build de produção com um WebXR falso, do botão à exportação
```

`verify` e `e2e` usam uma cena analítica (caixa 14 × 12 × 10 cm e esfera de 5 cm sobre uma mesa,
texturas procedurais sólidas) renderizada por ray casting em GLSL, com a mesma cena em JavaScript
para dar a verdade: profundidade, distância à superfície e cor. O `e2e` injeta um runtime WebXR
falso (`tools/fake-xr.ts`) que imita uma pessoa dando duas voltas em alturas diferentes, com
deriva de odometria, imagem bruta da câmera e profundidade do "sensor".

## Interface

- **Início** — botão *Start scan*; se o navegador não tiver AR com acesso à câmera ou WebGPU, a
  mensagem diz exatamente o que falta.
- **Captura (AR)** — a câmera ocupa a tela; por cima, a nuvem que está sendo triangulada ao vivo e
  os frustums dos quadros-chave já tirados, ancorados no mundo. Chips no topo: tempo de gravação,
  quadros-chave (e quantos foram descartados por borrão), pontos esparsos e se há profundidade do
  sensor. Um balão avisa *slow down* quando o movimento borraria a imagem e quando o rastreamento
  está limitado. O disparador grava/para; ✕ sai sem construir.
- **Pipeline** — ao parar, abre o visualizador e um cartão com as 8 etapas do pipeline: a etapa
  atual com barra de progresso, contador (ex. `37/119`) e cronômetro; as concluídas com o tempo que
  levaram; no rodapé o tempo de processamento e o total com a captura. Quando termina, o cartão
  recolhe numa linha (*Model built · 8/8 · 47.2 s*) e reabre com um toque.
- **Visualizador** — o modelo vai aparecendo enquanto é construído: primeiro a nuvem esparsa e as
  câmeras, depois a densa, a malha sombreada e por fim a texturizada. Um dedo gira, dois dedos dão
  zoom e deslocam, toque duplo reenquadra. Embaixo: câmeras/trajetória liga-desliga, camada
  *Mesh / Dense / Sparse* e a **régua** — dois toques na malha dão a distância em mm/cm (para o erro
  métrico |L̂ − L| do relatório).
- **ⓘ Diagnóstico** — quadros-chave, tamanho das imagens, comprimento do trajeto e distância início
  ↔ fim (deriva do laço), pontos e observações, **erro de reprojeção média / mediana** com as poses
  do celular e após o bundle adjustment, correção das poses, pontos densos e **densidade
  (pontos/m²)**, voxel, triângulos, área, dimensões do modelo, atlas, faces sem textura, GPU e o
  tempo de cada etapa.
- **⬇ Exportar** — `GLB` (malha texturizada + frustums das câmeras + nuvem esparsa, abre no Blender,
  MeshLab, three.js), `PLY` da nuvem densa (com normais e cor), `PLY` da esparsa e `JSON` com todas
  as métricas.

## O pipeline

Um único pipeline, sem configurações: os parâmetros abaixo são fixos ou derivados da própria
captura (profundidade da cena, distância focal, orçamento de memória).

| # | etapa | onde | o que faz |
|---|---|---|---|
| 1 | Captura e rastreamento | main + worker | quadros-chave, ORB, emparelhamento guiado, triangulação ao vivo |
| 2 | Bundle adjustment | worker (CPU) | refina poses e pontos com a odometria como priori |
| 3 | Estéreo denso | WebGPU | PatchMatch fotométrico por quadro-chave |
| 4 | Refinamento geométrico | WebGPU | PatchMatch com consistência entre mapas de profundidade |
| 5 | Fusão de profundidade | WebGPU + CPU | filtro de consistência multivista → nuvem densa |
| 6 | Malha | WebGPU + CPU | volume TSDF esparso → surface nets → limpeza |
| 7 | Seleção de vistas | WebGPU + CPU | visibilidade e melhor câmera por face |
| 8 | Atlas de textura | CPU | nivelamento de cor, costuras, empacotamento, JPEG |

### 1. Captura e rastreamento ao vivo

- **Sessão WebXR** `immersive-ar` com `camera-access` (imagem bruta da câmera via
  `XRWebGLBinding.getCameraImage`), `dom-overlay` e `depth-sensing` opcional. A pose de cada quadro
  vem de `view.transform` (câmera → mundo, espaço `local`, y para cima) e os intrínsecos da matriz
  de projeção: `fx = P₀₀·W/2`, `fy = P₁₁·H/2`, `cx = (1 − P₂₀)·W/2`, `cy = (1 + P₂₁)·H/2`.
- **Quadros-chave**: um novo quando, contra todos os anteriores, a vista mudou o bastante —
  `max(|ΔC| / (0,1·D), Δθ / 6°) ≥ 1`, com `D` a profundidade mediana da cena (≈ 6° entre vistas
  vizinhas, a faixa boa para MVS) — e a câmera está devagar (< 0,5 rad/s e < 0,35 m/s). No máximo
  120 por captura e um a cada 150 ms.
- **Cópia da imagem sem travar o AR**: a textura da câmera é desenhada numa FBO 1280 px (lado
  maior), lida num *pixel pack buffer* com `fenceSync` e recolhida quadros depois. A profundidade do
  sensor é amostrada numa grade 64 px (`getDepthInMeters`).
- **Rejeição de borrão** no worker: variância do Laplaciano abaixo de 50 % da mediana dos últimos
  10 quadros aceitos descarta o quadro.
- **Características ORB próprias**: FAST-9 em 3 níveis (fator 1,4), ranking por Harris (janela
  7×7, k = 0,04), supressão de não-máximos e distribuição em grade (células de 24 px), orientação
  pelo centróide de intensidade e descritor rBRIEF de 256 bits (padrão gaussiano fixo, σ = 31/5),
  1600 pontos por quadro em 640 × 360.
- **Emparelhamento guiado**: como as poses já são conhecidas, cada característica só é comparada
  com as da outra imagem dentro de uma faixa em torno do seu **segmento epipolar** (F calculada das
  poses, profundidade entre 5 cm e 15 m), em células de 16 px. Teste da razão de Lowe (0,82),
  Hamming ≤ 72 e verificação um-para-um. Pares vizinhos no tempo usam faixa de 4 px; pares de
  fechamento de laço (> 5 s de distância) usam 40 px — a deriva do celular — seguidos de **RANSAC
  do 8 pontos normalizado** em F (limiar de Sampson 2 px).
- **Trilhas e triangulação**: união das correspondências em trilhas (com fusão quando um laço liga
  duas trilhas), triangulação multivista por interseção de raios em mínimos quadrados + Gauss-Newton
  no erro de reprojeção; aceita com erro ≤ 4 px e ângulo ≥ 1,5°.

### 2. Bundle adjustment com a odometria como priori

Levenberg-Marquardt sobre todas as poses (6 GL cada) e até 8000 pontos (os de mais vistas),
perda de Huber (1,5 px), complemento de Schur para eliminar os pontos e **gradiente conjugado
pré-condicionado por blocos 6×6** no sistema reduzido das câmeras. O movimento relativo entre
quadros-chave consecutivos medido pelo celular entra como fator gaussiano
(σ_R = 0,25° + 3 %, σ_t = 2 mm + 3 % do deslocamento) e a primeira câmera fixa o gauge: assim a
**escala métrica e a gravidade vêm do celular** e as observações corrigem a deriva. Duas rodadas
(15 + 10 iterações) com poda de observações acima de 4 px e depois 2,5 px e re-triangulação.

### 3–4. Estéreo multivista PatchMatch (WebGPU)

Para cada quadro-chave de referência, em 640 × 360:

- **vistas fonte**: as 5 que mais compartilham pontos esparsos, cada ponto pesando
  `exp(−(θ − 10°)² / 2σ²)` pelo ângulo de triangulação; intervalo de profundidade dos pontos
  vistos (percentis 2–98 % × 0,6/1,5) e da profundidade do sensor;
- **hipótese por pixel** = plano (normal, profundidade); custo = 1 − NCC com pesos bilaterais
  (σ_cor = 0,06, σ_espaço = 5 px) numa janela 11 × 11 amostrada a cada 2 px, mapeada pela
  homografia do plano `H = K_s (R + t nᵀ/ρ) K_r⁻¹`; custos das vistas agregados pela média das
  3 melhores (robusto a oclusão);
- **propagação em xadrez vermelho/preto adaptativa** (ACMH): em 8 regiões em volta do pixel escolhe
  o vizinho de menor custo atual e transfere o plano dele;
- **refinamento** com hipóteses aleatórias e perturbadas (passo × 0,5 por iteração), 5 iterações;
- a profundidade do sensor, quando existe, **concorre** com a hipótese aleatória na inicialização
  (não a substitui: em superfícies com textura a propagação acha planos melhores; nas lisas ela
  sobrevive);
- **passada geométrica** (ACMM): mais 2 iterações somando 0,2 × erro de reprojeção ida-e-volta
  (limitado a 3 px) contra os mapas das vistas fonte.

Todas as imagens cinza ficam numa `texture_2d_array` e todos os mapas num buffer único, empacotados
em 32 bits por pixel (profundidade f16 | normal octaédrica 6+6 bits | custo 4 bits).

### 5. Fusão de profundidade → nuvem densa

Cada profundidade é reprojetada nas 8 melhores vistas, lida no mapa delas e trazida de volta: ela
fica se **≥ 40 % das vistas** (mínimo 2) concordarem em menos de 1 px e 1 % de profundidade, e vira
a média das concordantes. Isso elimina o "engorde" de bordas (profundidade que vaza pela silhueta).
Os pontos são acumulados numa grade hash de voxels (posição, normal e cor médias): é a **nuvem
densa**, e o tamanho da célula — `1,6 × profundidade mediana / f`, crescendo 1,4× enquanto passar
de 500 mil células (malhas de até ~1 milhão de triângulos) — vira o voxel da malha.

### 6. Malha: TSDF esparso + surface nets

Volume TSDF em blocos 8³ alocados só perto da superfície (pontos densos e profundidade do sensor),
integrado na GPU mapa a mapa (truncamento 4 voxels, peso pelo ângulo de visão, a profundidade do
sensor preenche buracos com peso 0,12). A superfície sai por **surface nets** (um vértice por célula
que cruza o zero, um quadrilátero por aresta, células com peso < 0,6 ignoradas — nada nasce do
espaço não observado) e componentes com menos de 1 % das faces são removidos.

### 7–8. Textura

- **Visibilidade**: o z-buffer de cada vista é renderizado na GPU; uma face entra como candidata se
  está de frente, cai dentro da imagem e não está atrás do z-buffer no centróide. Nota = área
  projetada em pixels × nitidez da vista. Cada face guarda as 4 melhores.
- **Charts por crescimento**: começando da face com melhor nota, um chart absorve vizinhas enquanto
  a vista dele valer ≥ 60 % da melhor de cada vizinha; charts minúsculos são absorvidos.
- **Nivelamento de cor**: ganho RGB por vista em mínimos quadrados no domínio log (Huber) com as
  faces vistas por várias câmeras; nas costuras, as cores são puxadas para a média e a correção é
  espalhada pelo chart como campo harmônico (à la Waechter et al.).
- **Atlas**: os UVs de cada chart são a própria projeção na vista; prateleiras empacotadas até
  4096 px, recortes copiados com interpolação bilinear e borda de 2 px dilatada, salvo em JPEG.

## Validação (cena sintética, RTX 4070 Laptop)

`npm run verify` — 60 quadros 1280 × 720, deriva simulada de 1,8 cm/0,9° ao longo da captura:

| verificação | resultado |
|---|---|
| PatchMatch nas poses verdadeiras | 100 % dos pixels de superfície, erro relativo mediano 0,04 % |
| bundle adjustment | erro dos centros 11,9 → 1,3 mm (0,6 mm após alinhamento rígido) |
| reprojeção | média 0,65 px, mediana 0,50 px |
| nuvem densa | erro mediano 0,31 mm, p90 1,32 mm (voxel 2,1 mm) |
| malha | erro mediano 0,29 mm, p90 0,70 mm, completude 100 % (≤ 3 mm) |
| dimensões da caixa (faces, como um paquímetro) | 140,0 × 119,8 × 99,9 mm vs 140 × 120 × 100 mm |
| textura | erro de cor mediano 1,3 / 255, 4,6 mil charts |

`npm run e2e` — o app real, 33 s de captura (119 quadros-chave), processamento em ≈ 14 s; o GLB
exportado fica a 0,45 mm (mediana) da superfície verdadeira. Num celular topo de linha a estimativa
(não medida) é de 1–2 minutos de processamento para uma captura de 30–60 s, pela razão de
desempenho de GPU (~6×) e CPU (~3×) em relação a este notebook.

## Relação com o trabalho

| etapa do enunciado | no app |
|---|---|
| 2 — detecção/descrição | ORB próprio (FAST + Harris + rBRIEF), 3 escalas |
| 3 — emparelhamento e geometria epipolar | razão de Lowe + um-para-um, faixa epipolar das poses, RANSAC em F nos laços |
| 4 — SfM, escala, MVS, malha, textura | triangulação + BA (escala métrica do ARCore), PatchMatch, TSDF + surface nets, atlas |
| 5 — métricas | reprojeção média/mediana, pontos esparsos/densos, densidade, régua para \|L̂ − L\|, deriva do laço |
| extras | poses das câmeras no espaço 3D, exportação GLB/PLY |

A comparação entre detectores, a referência de escala impressa e os estudos de ablação ficam para
o pipeline do relatório; aqui a escala vem da odometria do celular e pode ser conferida com a régua.

## Limitações

- Só Chrome/Android com ARCore (WebXR com acesso à câmera) e WebGPU.
- Superfícies sem textura, reflexivas ou transparentes não dão estéreo; com sensor de profundidade
  elas ainda entram na malha com o peso baixo do sensor, sem ele ficam buracos.
- A orientação do celular deve ser a mesma do início ao fim (quadros noutra orientação são
  ignorados).
- Não há modelo de distorção da lente: os intrínsecos do ARCore são usados como estão (só pinhole).
- Ainda não foi testado num celular real — a captura WebXR foi validada com o runtime falso do
  `e2e`; o primeiro teste no aparelho é o próximo passo.

## Código

```
src/app/core/        serviços (WebXR, ponte com o worker), modelos, worker, tipos WebXR
src/app/render/      renderizador WebGL2 (pontos, linhas, malha) e câmera orbital
src/app/ui/          captura, visualizador, pipeline, diagnóstico, exportação
src/app/vision/
  math/ geometry/    SO(3), Cholesky, Jacobi, câmera, triangulação
  image/ features/   cinza, nitidez, ORB
  matching/ sfm/     emparelhamento guiado, RANSAC, trilhas, bundle adjustment
  gpu/ mvs/          WebGPU, PatchMatch (WGSL), seleção de vistas
  fusion/            nuvem densa, TSDF (WGSL), surface nets
  texture/ export/   visibilidade, charts, atlas, GLB, PLY
  pipeline/          sessão de captura e reconstrução
tools/               cena sintética, verify, WebXR falso, e2e
```
