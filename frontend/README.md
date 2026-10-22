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
npm test -- --watch=false           # 35 testes unitários (Vitest)
npm run typecheck                   # app, worker, testes e ferramentas
npm run verify                      # pipeline inteiro numa cena sintética, Chrome headless + WebGPU
npm run e2e                         # o build de produção com um WebXR falso, do botão à exportação
```

`verify` e `e2e` usam uma cena analítica (caixa 14 × 12 × 10 cm e esfera de 5 cm sobre uma mesa,
texturas procedurais sólidas) renderizada por ray casting em GLSL, com a mesma cena em JavaScript
para dar a verdade: profundidade, distância (com sinal) à superfície e cor. O `e2e` injeta um
runtime WebXR falso (`tools/fake-xr.ts`) que imita uma pessoa dando duas voltas em alturas
diferentes, com deriva de odometria, imagem bruta da câmera e profundidade do "sensor".
`VERIFY_QUERY='?quick&vary=stereo.sources:6,8;texture:levelled,mrf' npm run verify` compara
variantes do High (qualquer campo do perfil, por caminho), `&truth` reconstrói com as poses
verdadeiras (o piso do pipeline).

## Interface

- **Início** — seletor de qualidade **Low / Medium / High** (High é o padrão; a escolha fica
  salva) com uma linha do que muda, e o botão *Start scan*; se o navegador não tiver AR com acesso
  à câmera ou WebGPU, a mensagem diz exatamente o que falta.
- **Captura (AR)** — a câmera ocupa a tela; por cima, a nuvem que está sendo triangulada ao vivo e
  os frustums dos quadros-chave já tirados, ancorados no mundo. Chips no topo: tempo de gravação,
  quadros-chave (e quantos foram descartados por borrão), pontos esparsos e se há profundidade do
  sensor. Um balão avisa *slow down* quando um quadro-chave está atrasado por causa do movimento,
  quando o rastreamento está limitado e quando o orçamento de quadros-chave encheu. O disparador
  grava/para; ✕ sai sem construir.
- **Pipeline** — ao parar, abre o visualizador e um cartão com as 8 etapas: em cada uma o
  **método usado naquela qualidade** (ex. *SIFT on the GPU · 20k points · focal self-calibration*),
  a etapa atual com barra de progresso, contador (ex. `37/119`) e cronômetro, as concluídas com o
  tempo que levaram; no rodapé o tempo de processamento e o total com a captura. Quando termina, o
  cartão recolhe numa linha (*Model built · HIGH · 8/8 · 1:16*) e, aberto, oferece **Rebuild** em
  outra qualidade a partir da mesma captura (também depois de uma falha).
- **Visualizador** — o modelo vai aparecendo enquanto é construído: primeiro a nuvem esparsa e as
  câmeras, depois a densa, a malha sombreada e por fim a texturizada (ou com cor por vértice no
  Low). Um dedo gira, dois dedos dão zoom e deslocam, toque duplo reenquadra. Embaixo:
  câmeras/trajetória liga-desliga, camada *Mesh / Dense / Sparse* e a **régua** — dois toques na
  malha dão a distância em mm/cm (para o erro métrico |L̂ − L| do relatório).
- **ⓘ Diagnóstico** — qualidade, quadros-chave (capturados e usados no denso), rastreamento
  (quebras, pares resgatados, segmentos re-ancorados), correção da focal, tamanho do estéreo,
  comprimento do trajeto e distância início ↔ fim, pontos e observações, **erro de reprojeção
  média / mediana** antes e depois do bundle adjustment, pontos densos e **densidade (pontos/m²)**,
  voxel, triângulos, área, dimensões, textura, GPU e o tempo de cada etapa.
- **⬇ Exportar** — `GLB` (malha texturizada ou com `COLOR_0` + frustums das câmeras + nuvem
  esparsa, abre no Blender, MeshLab, three.js), `PLY` da nuvem densa (com normais e cor), `PLY` da
  esparsa e `JSON` com todas as métricas.

## O pipeline

As mesmas 8 etapas em todas as qualidades; o que muda é o **método** de cada uma, escolhido pelo
perfil de qualidade (`core/models/quality.ts`). Os métodos são peças trocáveis registradas por id
— características em `vision/features/methods.ts`, estéreo em `vision/mvs/methods/`, textura em
`vision/texture/methods/` — e o orquestrador (`vision/pipeline/reconstructor.ts`) só os encadeia.

| # | etapa | Low | Medium | High |
|---|---|---|---|---|
| 1 | Captura e rastreamento | ORB, emparelhamento guiado, triangulação ao vivo | = | = |
| 2 | Bundle adjustment | ORB ao vivo, 1 rodada, 3 mil pontos | ORB ao vivo, 2 rodadas, 8 mil pontos | **SIFT na GPU** re-extraído, 20 mil pontos, **auto-calibração da focal** |
| 3 | Estéreo denso | PatchMatch ½ resolução, 3 iterações, 4 vistas | PatchMatch 640 px, 5 iterações, 5 vistas | **coarse-to-fine** ½ → **1,5×** (960 px), 6 + 4 iterações, **8 vistas** |
| 4 | Refinamento geométrico | — | ACMM, 1 iteração | ACMM, 3 iterações |
| 5 | Fusão de profundidade | 2 de 4 vistas | 40 % de 8 vistas | 40 % de 8 vistas (malha: **2 de 8**) |
| 6 | Malha | TSDF, voxel grosso | TSDF | TSDF **ponderado por confiança**, voxel fino |
| 7 | Seleção de vistas | z-buffer ¼ | crescimento de charts, z-buffer ½ | **MRF** (qualidade × costuras), z-buffer inteiro |
| 8 | Textura | **cor por vértice** | atlas com ganhos de exposição | atlas + **nivelamento de costuras** |
| | quadros-chave no denso | até 40 | até 80 | até 150 |

Toda reconstrução parte da captura como foi gravada (o mapa é salvo antes do primeiro build e
restaurado a cada um), então reconstruir em outra qualidade é repetível.

### 1. Captura e rastreamento ao vivo

- **Sessão WebXR** `immersive-ar` com `camera-access` (imagem bruta da câmera via
  `XRWebGLBinding.getCameraImage`), `dom-overlay` e `depth-sensing` opcional. A pose de cada quadro
  vem de `view.transform` (câmera → mundo, espaço `local`, y para cima) e os intrínsecos da matriz
  de projeção: `fx = P₀₀·W/2`, `fy = P₁₁·H/2`, `cx = (1 − P₂₀)·W/2`, `cy = (1 + P₂₁)·H/2`.
- **Quadros-chave**: um novo quando, contra todos os anteriores, a vista mudou o bastante —
  `max(|ΔC| / (0,1·D), Δθ / 6°) ≥ 1`, com `D` a profundidade mediana da cena — e a câmera está
  devagar (< 0,25 rad/s e < 0,2 m/s); se a vista já mudou 1,6× o espaçamento, aceita até
  0,5 rad/s e 0,35 m/s. No máximo 180 por captura e um a cada 150 ms; cada qualidade usa depois o
  seu subconjunto (descarta primeiro os redundantes e os borrados, mantendo a cobertura).
- **Cópia da imagem sem travar o AR**: a textura da câmera é desenhada numa FBO 1280 px (lado
  maior), lida num *pixel pack buffer* com `fenceSync` e recolhida quadros depois. A profundidade do
  sensor é amostrada numa grade 64 px (`getDepthInMeters`).
- **Rejeição de borrão** no worker: variância do Laplaciano abaixo de 50 % da mediana dos últimos
  10 quadros aceitos descarta o quadro.
- **Características ORB próprias**: FAST-9 em 3 níveis (fator 1,4), ranking por Harris, distribuição
  em grade (células de 24 px), orientação pelo centróide de intensidade e rBRIEF de 256 bits,
  1600 pontos por quadro em 640 × 360 — rápido o bastante para todo quadro-chave no celular.
- **Emparelhamento guiado**: cada característica só é comparada com as da outra imagem numa faixa
  em torno do seu **segmento epipolar** (F das poses), em células de 16 px. Razão de Lowe (0,82),
  limite de distância do descritor e um-para-um. Pares vizinhos no tempo usam faixa de 4 px; pares
  de fechamento de laço (> 5 s de distância) usam 40 px — a deriva do celular — seguidos de **RANSAC
  do 8 pontos normalizado** em F (limiar de Sampson 2 px).
- **Trilhas e triangulação**: união das correspondências em trilhas (com fusão quando um laço liga
  duas trilhas), triangulação multivista por mínimos quadrados + Gauss-Newton no erro de
  reprojeção; aceita com erro ≤ 4 px e ângulo ≥ 1,5°.
- **Robustez a saltos de rastreamento** (`pipeline/capture/pose-guard.ts`):
  - perda de rastreamento informada pelo celular ou movimento fisicamente implausível (> 1,5 m/s,
    > 3 rad/s) quebra o elo de odometria entre os dois quadros;
  - se o par consecutivo de repente tem bem menos correspondências guiadas que os anteriores, ele é
    refeito **sem as poses** (força bruta + RANSAC em F); se as imagens claramente se sobrepõem, a
    pose do celular é que está errada;
  - o quadro novo é conferido contra os pontos que o mapa já tem: um **pico** do erro de reprojeção
    (> 2,5 px e > 3× o usual — um salto, mesmo ao longo das epipolares, que o emparelhamento guiado
    não vê) leva a **ressecção** (LM com Huber e poda) e o resto daquele segmento de rastreamento
    segue a correção.

### 2. Bundle adjustment com a odometria como priori

Levenberg-Marquardt sobre todas as poses (6 GL cada), os pontos de mais vistas e, no High, **um
fator de escala da focal comum a todos os quadros** (priori fraca, σ = 5 %), perda de Huber
(1,5 px), complemento de Schur para eliminar os pontos e **gradiente conjugado pré-condicionado por
blocos** no sistema reduzido. O movimento relativo entre quadros-chave consecutivos medido pelo
celular entra como fator **robusto** (Huber a 3σ; σ_R = 0,25° + 3 %, σ_t = 2 mm + 3 % do
deslocamento), exceto através das quebras: a **escala métrica e a gravidade vêm do celular**, as
observações corrigem a deriva e um salto não arrasta a solução. Rodadas de 15/12/10 iterações com
poda de observações acima de 6, 4 e 2,5 px e re-triangulação; antes da primeira, trilhas que a
deriva partiu (laços) ganham um ponto da parte consistente.

**High — SIFT re-extraído** (`features/sift/`): depois de uma rodada no mapa ORB (que já tira a
deriva e o erro da focal), cada quadro-chave passa por um **SIFT completo na GPU** — imagem ampliada
2× (Lowe), espaço de escala gaussiano de 3 intervalos por oitava, extremos do DoG refinados a
subpixel por ajuste quadrático, rejeição de baixo contraste e de bordas, orientação dominante e
descritor 4 × 4 × 8 com interpolação trilinear, **RootSIFT** quantizado em bytes. 2000 pontos por
quadro em ≈ 10 ms (a versão de referência em CPU, usada nos testes, leva ≈ 0,5 s). As trilhas são
refeitas com faixas epipolares de 4 px das poses já refinadas, limitadas ao intervalo de
profundidade de cada quadro, e o bundle adjustment roda de novo nelas; as poses voltam para o mapa
ao vivo. Por que: o ORB localiza os cantos no pixel inteiro do seu nível da pirâmide (erro de
0,5–0,7 px), o SIFT a ~0,02 px; na cena sintética isso reduz o erro de rotação relativa entre
quadros de 65 para 20 milésimos de grau e o erro de forma da malha à metade.

### 3–4. Estéreo multivista PatchMatch (WebGPU)

Para cada quadro-chave de referência:

- **vistas fonte**: as que mais compartilham pontos esparsos, cada ponto pesando
  `exp(−(θ − 10°)² / 2σ²)` pelo ângulo de triangulação; intervalo de profundidade dos pontos
  vistos (percentis 2–98 % × 0,6/1,5) e da profundidade do sensor;
- **hipótese por pixel** = plano (normal, profundidade); custo = 1 − NCC com pesos bilaterais
  (σ_cor = 0,06, σ_espaço = 5 px) numa janela 11 × 11 amostrada a cada 2 px, mapeada pela
  homografia do plano `H = K_s (R + t nᵀ/ρ) K_r⁻¹`; custos das vistas agregados pela média das
  melhores (robusto a oclusão);
- **propagação em xadrez vermelho/preto adaptativa** (ACMH): em 8 regiões em volta do pixel escolhe
  o vizinho de menor custo atual e transfere o plano dele;
- **refinamento** com hipóteses aleatórias e perturbadas (passo × 0,5 por iteração);
- a profundidade do sensor, quando existe, **concorre** com a hipótese aleatória na inicialização;
- **coarse-to-fine** (High): resolve primeiro em meia resolução (a janela cobre o dobro da cena,
  ajuda em regiões pouco texturizadas) e começa a resolução final pelos planos de lá;
- **resolução**: High resolve a **1,5×** das imagens de trabalho (960 × 540, decodificadas dos
  JPEG de 1280 px dos quadros-chave, um de cada vez para não acumular memória) enquanto a GPU couber
  todas as vistas; senão desce até 640 px. A reamostragem por área usa pesos fracionários, para que
  os centros dos pixels fiquem exatamente onde os intrínsecos escalados dizem;
- **passada geométrica** (ACMM): iterações somando 0,2 × erro de reprojeção ida-e-volta (limitado a
  3 px) contra os mapas das vistas fonte.

Todas as imagens cinza ficam numa `texture_2d_array` e todos os mapas num buffer único, empacotados
em 32 bits por pixel (profundidade f16 | normal octaédrica 6+6 bits | custo 4 bits). O trabalho é
fatiado em faixas de linhas com altura adaptativa (~20 ms por submit) para não travar a tela nem
disparar o watchdog da GPU do celular.

### 5. Fusão de profundidade → nuvem densa

Cada profundidade é reprojetada nas vistas de checagem, lida no mapa delas e trazida de volta: ela
fica se a fração exigida das vistas concordar em menos de 1 px e 1 % de profundidade, e vira a
média das concordantes. Isso elimina o "engorde" de bordas. No High a malha usa um filtro mais
frouxo (2 de 8 vistas) que a nuvem densa: o TSDF tira a média de muitas profundidades e ainda pesa
cada uma pela confiança, então aproveita superfície que o filtro estrito jogaria fora. Os pontos são acumulados numa grade
hash de voxels (posição, normal e cor médias): é a **nuvem densa**, e o tamanho da célula —
`fator × profundidade mediana / f`, crescendo enquanto passar do orçamento de células da
qualidade — vira o voxel da malha.

### 6. Malha: TSDF esparso + surface nets

Volume TSDF em blocos 8³ alocados só perto da superfície (pontos densos e profundidade do sensor),
integrado na GPU mapa a mapa (truncamento 4 voxels, a profundidade do sensor preenche buracos com
peso 0,12). O peso de cada amostra é o ângulo de visão; no High também a **confiança**:
`(1 − custo)²` do PatchMatch e `(D/d)²` (o ruído do estéreo cresce com d², então vistas mais
próximas valem mais). A superfície sai por **surface nets** (um vértice por célula que cruza o zero,
células com peso < 0,6 ignoradas — nada nasce do espaço não observado) e componentes pequenos são
removidos.

### 7–8. Textura

- **Visibilidade**: o z-buffer de cada vista é renderizado na GPU (na resolução da qualidade); uma
  face entra como candidata se está de frente, cai dentro da imagem e não está atrás do z-buffer.
  Nota = área projetada × nitidez da vista. Cada face guarda as melhores.
- **Low — cor por vértice**: cada vértice mistura as cores das duas melhores vistas (miniaturas),
  sem atlas.
- **Charts por crescimento** (Medium): começando da face com melhor nota, um chart absorve
  vizinhas enquanto a vista dele valer ≥ 60 % da melhor de cada vizinha.
- **MRF** (High, `texture/mrf-labeling.ts`, à la Lempitsky & Ivanov / Waechter et al.): cada face
  paga `1 − nota/melhor nota` pela vista escolhida e cada aresta entre faces de vistas diferentes
  paga 1 (uma costura); minimizado sobre as 4 vistas candidatas de cada face por *loopy belief
  propagation* min-sum (mensagens de Potts amortecidas) e polido por ICM. O crescimento de charts
  também é polido e fica a solução de menor energia, então nunca é pior que ele.
- **Nivelamento de cor**: ganho RGB por vista em mínimos quadrados no domínio log (Huber); no High
  as cores das costuras são puxadas para a média e a correção é espalhada pelo chart como campo
  harmônico (à la Waechter et al.).
- **Atlas**: os UVs de cada chart são a própria projeção na vista; prateleiras empacotadas até
  4096 px, recortes copiados com interpolação bilinear e borda dilatada, salvo em JPEG.

## Validação (cena sintética, RTX 4070 Laptop)

`npm run verify` — 52 verificações, todas passando. 60 quadros 1280 × 720, deriva simulada que
chega a 2,3 cm / 0,9° no fim da captura. Além do erro da malha depois de alinhar os centros das
câmeras à verdade, o verify mede o **erro de forma**: a malha ajustada rigidamente à superfície
verdadeira, o que separa a qualidade do modelo de onde ele fica no gauge.

| captura "orbit" | Low | Medium | High |
|---|---|---|---|
| processamento | 3,4 s | 11,9 s | 40,7 s |
| reprojeção (mediana) | 0,50 px | 0,50 px | 0,20 px |
| centros das câmeras após alinhamento rígido | 0,77 mm | 0,62 mm | 0,34 mm |
| rotação relativa entre quadros (mediana) | 71 m° | 67 m° | 21 m° |
| malha, erro de forma mediano / p90 | 0,23 / 0,86 mm | 0,13 / 0,71 mm | 0,04 / 0,18 mm |
| caixa (faces, como um paquímetro) | 139,7 × 120,0 × 99,7 | 140,0 × 119,8 × 99,9 | 139,9 × 119,9 × 100,0 |
| textura: degrau de cor nas costuras (p90, 0–255) | cor por vértice | 6,7 | 5,3 |

Com as poses verdadeiras o erro de forma é 0,03 / 0,15 mm. Cada etapa do High teve alternativas
medidas isoladamente nas capturas "orbit" e "hard" (erro de forma mediano orbit / hard):

| etapa | alternativas testadas | resultado | escolha |
|---|---|---|---|
| captura | ORB ao vivo | SIFT ao vivo dividiria a GPU com a câmera do AR (a sobrecarga da GPU já travou a interface do celular); o High re-extrai SIFT depois | ORB |
| bundle adjustment | ORB → **SIFT** | 0,126 → 0,056 / 0,160 → 0,067 mm; rotação relativa 65 → 20 m° | SIFT (+8 s) |
| estéreo, resolução | 640 → **960 px** | 0,056 → 0,048 / 0,067 → 0,057; p90 0,60 → 0,36 | 960 px (+8 s) |
| estéreo, vistas fonte | 6 / **8** / 10 / 12 | 0,048 → 0,041 / 0,065 → 0,048 com 8; 10 e 12 quase nada a mais e nuvem densa mais suja | 8 (+4 s) |
| estéreo, iterações | 6 / 9 | igual | 6 |
| refinamento geométrico | 1 / **3** / 5 iterações | 0,050 / 0,051 / 0,053 — indistinguível nesta cena toda texturizada (o ganho do ACMM é em regiões pobres de textura) | 3 |
| fusão, concordância | 2, 3, 4, 6 de 8 vistas | exigir mais limpa a nuvem densa (p90 0,71 → 0,42 mm com 4) mas piora a malha (0,051 → 0,057); exigir 2 suja a nuvem (p90 2,35 mm) e melhora a malha (voxel fixo: 0,051 → 0,048 / 0,068 → 0,065, p90 −10 %) | 3 na nuvem, **2 na malha** |
| malha, peso | ângulo / **confiança** | p90 0,36 → 0,27, mediana +3–8 µm | confiança |
| malha, resolução | 600 mil / 1 milhão de células | p90 −8 a −17 %, mediana igual; nuvem densa 2× e malha 13–40 % maiores | 600 mil (memória do celular) |
| seleção de vistas | crescimento de charts / **MRF** | charts −25 a −40 %, degrau nas costuras p90 6,0 → 5,0 com vinheta, mesma cor | MRF (+1 s) |
| textura | sem / **com** nivelamento de costuras | com vinheta: degrau p90 15,3 → 6,0 | com |

Outras verificações: PatchMatch nas poses verdadeiras (100 % dos pixels, erro relativo mediano
0,04 %); SIFT da GPU igual ao de referência em CPU (99,9 % dos pontos, 0,0001 px, descritores a
2/512); reconstruir High depois das outras dá o mesmo modelo; **"hard"** (focal informada 2 % errada, um
salto silencioso de 4 cm / 1,5° no quadro 18, exposição automática variando ±15 % e vinheta de
25 % nos cantos): salto detectado e re-ancorado, focal corrigida em −1,97 % (verdade −1,96 %), caixa
com 0,2 mm de erro e forma 0,04 mm no High contra 3,1 mm e 0,39 mm no Medium (sem
auto-calibração), degrau de cor nas costuras p90 5,0 contra 16,7; **"lost"** (perda de
rastreamento com salto, informada): quebra no quadro 42, segmento re-ancorado, forma 0,05 mm. As
cores são comparadas depois de um ganho global por canal (o brilho absoluto não é recuperável).

`npm run e2e` — o app real em 20 verificações: 33 s de captura (77 quadros-chave), High em 48 s
(GLB a 0,22 mm mediana da superfície verdadeira), depois **Rebuild** em Medium (16 s, 0,49 mm) e Low
(5 s, 0,67 mm, cor por vértice), conferindo os métodos mostrados em cada etapa. Num celular topo
de linha a estimativa (não medida) é de algumas vezes isso — 2 a 4 minutos de processamento no
High para uma captura de 30–60 s, menos de 1 minuto no Low.

## Relação com o trabalho

| etapa do enunciado | no app |
|---|---|
| 2 — detecção/descrição | ORB próprio (FAST + Harris + rBRIEF) ao vivo; SIFT próprio (DoG + RootSIFT, GPU e CPU) no High — com a comparação medida acima |
| 3 — emparelhamento e geometria epipolar | razão de Lowe + um-para-um, faixa epipolar das poses, força bruta + RANSAC em F quando a pose falha |
| 4 — SfM, escala, MVS, malha, textura | triangulação + BA (escala métrica do ARCore, focal auto-calibrada), PatchMatch, TSDF + surface nets, atlas |
| 5 — métricas | reprojeção média/mediana, pontos esparsos/densos, densidade, régua para \|L̂ − L\|, deriva do laço |
| extras | poses das câmeras no espaço 3D, três qualidades com métodos trocáveis, exportação GLB/PLY |

A referência de escala impressa e os estudos de ablação completos ficam para o pipeline do
relatório; aqui a escala vem da odometria do celular e pode ser conferida com a régua.

## Limitações

- Só Chrome/Android com ARCore (WebXR com acesso à câmera) e WebGPU.
- Superfícies sem textura, reflexivas ou transparentes não dão estéreo; com sensor de profundidade
  elas ainda entram na malha com o peso baixo do sensor, sem ele ficam buracos.
- A orientação do celular deve ser a mesma do início ao fim (quadros noutra orientação são
  ignorados).
- Não há modelo de distorção da lente: os intrínsecos do ARCore são usados como estão (pinhole); o
  High corrige só a escala da focal.
- Os tempos no celular não foram medidos: o High ficou ~2,3× mais lento que antes das trocas de
  método (SIFT, 960 px, 8 vistas, MRF), e celulares com limite de buffer da GPU menor usam menos
  resolução no estéreo.
- A cena sintética é toda texturizada: alternativas que só ajudam em superfícies lisas (mais
  iterações geométricas, priors planares do ACMMP) não puderam ser medidas aqui.

## Código

```
src/app/core/
  models/            perfis de qualidade, etapas, protocolo do worker, métricas
  services/          captura WebXR, ponte com o worker
  xr/                cópia assíncrona da câmera, política de quadros-chave, profundidade
  workers/           worker de reconstrução
src/app/render/      renderizador WebGL2 (pontos, linhas, malha) e câmera orbital
src/app/ui/          captura, visualizador, pipeline, diagnóstico, exportação
src/app/vision/
  math/ geometry/    SO(3), PCG por blocos, Jacobi, câmera, triangulação, ressecção
  image/             cinza, reamostragem por área, nitidez
  features/          FeatureSet + métrica do descritor, ORB, SIFT (CPU e GPU), registro de métodos
  matching/          emparelhamento guiado, força bruta, RANSAC em F
  sfm/               mapa esparso, ligação de trilhas, bundle adjustment (sfm/bundle/)
  gpu/ mvs/          WebGPU, PatchMatch (WGSL), seleção de vistas, métodos de estéreo
  fusion/            nuvem densa, TSDF (WGSL), surface nets, malha
  texture/ export/   visibilidade, charts, atlas, métodos de textura, GLB, PLY
  pipeline/          sessão de captura (capture/), etapas (stages/) e o orquestrador
tools/               cena sintética, verify (tools/verify/), WebXR falso, e2e
```
