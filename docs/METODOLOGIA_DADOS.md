# Metodologia de dados e experimentos

Texto base para a seção de procedimentos metodológicos. Os números do dataset vêm da execução de referência e são regenerados em `RELATORIO_DATASET.md` a cada treino.

## 1. Bases utilizadas

| Base | Origem | Conteúdo | Uso |
|---|---|---|---|
| Braille Detection v3 | Roboflow Universe (braille-lq5eh), CC BY 4.0 | Fotos com caixa por letra (A a Z) | Detecção |
| Braille Detection v1 | Roboflow Universe (projeto braille-detection-vxtp1), CC BY 4.0 | Mesmas fotos originais da v3 com outro tratamento | Detecção (agrupada com a v3) |
| Angelina Braille Images Dataset | Ovodov (2021, 2023a) | 212 páginas de livros, 28 manuscritas, 50 de teste e 44 fotos sem Braille, anotadas pelo código de 6 bits | Detecção, negativos e teste em foto real |
| Multi Braille Characters | Widyadara (2026), Mendeley Data | 6.606 fotos de pares de células em relevo com grade e 3.150 recortes de folhas | Classificador e páginas montadas |
| Braille Character Dataset | Kaggle (shanks0465) | 1.560 recortes (20 originais × 3 aumentos por letra) | Classificador e páginas montadas |
| Braille Character Image Classification | Kaggle | 2.600 recortes | Classificador e páginas montadas |
| Sintéticas | Geradas pelo projeto | Tinta e relevo simulado, com perspectiva, rotação, desfoque, ruído e JPEG | Detecção |
| Teste em português | Fotografado pelo grupo | Caixas de remédio (RDC Anvisa nº 71/2009) com o texto de referência digitado | Somente teste |

As bases de cenas naturais chinesas (LU et al., 2023; WU; LU; XIONG, 2025) foram solicitadas aos autores, sem resposta, e não foram utilizadas. O papel de representar fotos reais foi assumido pelo Angelina.

## 2. Padronização das classes

Todas as bases foram convertidas para 63 classes, uma por combinação dos seis pontos, usando o código binário de Ovodov (2021). A classe é o padrão de pontos, não a letra, o que torna o detector independente do idioma. A tradução para o português, com acentos, ç, sinal de maiúscula, sinal de número e pontuação, é feita depois, pela tabela da Grafia Braille para a Língua Portuguesa (BRASIL, 2018).

## 3. Cuidados contra vazamento entre treino e teste

1. As versões v1 e v3 do Roboflow contêm as mesmas 1.321 fotos originais. As imagens foram agrupadas pela foto de origem, e cada grupo foi inteiro para treino, validação ou teste.
2. No Multi Braille, as variações de brilho, exposição, matiz e saturação de uma mesma foto ficaram no mesmo grupo.
3. No Angelina foi mantida a divisão oficial dos autores; as 50 fotos enviadas por usuários formam o teste de foto real.
4. O teste em português não participa do treino.
5. Algumas fotos do Angelina têm apenas parte das celas anotadas. Como as celas não anotadas seriam aprendidas como fundo, blocos com menos de 15 celas anotadas foram descartados (cerca de 2% dos blocos).

## 4. Rotulagem do Multi Braille

As 105 pastas correspondem a sílabas formadas por 21 consoantes (b, c, d, f, g, h, j, k, l, m, n, p, q, r, s, t, v, w, x, y, z) e 5 vogais (a, i, u, e, o). A correspondência foi verificada detectando automaticamente o relevo em cada quadrado da grade, com concordância em 97 das 105 pastas. As linhas da grade foram removidas antes do uso. Quando a grade não é detectada, e nos recortes sem grade, a célula da direita só é aproveitada se tiver pontos nas duas colunas, porque a borda do recorte não permite determinar a posição da coluna em células como "a".

## 5. Composição do dataset

Na execução final: 1.500 imagens sintéticas e 1.500 páginas montadas, somadas às bases reais. A quantidade de sintéticas foi limitada para que as celas reais (Angelina e Roboflow) não fossem minoria, já que dados sintéticos tendem a perder desempenho em fotos reais (VIJAY et al., 2026). Os números exatos estão em `RELATORIO_DATASET.md`, com 62 das 63 classes representadas; a ausente e as mais raras são sinais de pontuação pouco frequentes em textos reais. Páginas grandes do Angelina são divididas em quatro blocos sobrepostos, para que a cela não fique pequena demais na entrada de 640 pixels.

## 6. Treinamento

| Parâmetro | Valor | Justificativa |
|---|---|---|
| imgsz | 640 (416 no teste de resolução) | Plano de testes da orientadora |
| epochs / patience | 60 / 30 | Orçamento de GPU do Colab |
| optimizer / lr0 / momentum | SGD / 0,01 / 0,937 | Fixos para comparações justas entre modelos |
| fliplr / flipud | 0 / 0 | Espelhar troca o ponto 1 pelo 4 e altera a letra |
| degrees | 5 | Rotação usada por Ovodov (2021) |
| iou (NMS) | 0,45 | As celas não se sobrepõem |
| seed | 42 | Reprodutibilidade |

## 7. Modelos comparados

1. YOLO11n com 63 classes, em 640 e 416 pixels
2. YOLO26n com 63 classes, mesmos hiperparâmetros
3. Duas etapas: YOLO11n de classe única localiza as celas e MobileNetV2 classifica cada recorte (SETIAWAN; RACHMAWANTO, 2025)
4. Método clássico por grade: detecção de pontos por limiarização, reconstrução geométrica da grade e agrupamento em celas

## 8. Avaliação

Cada domínio tem seu próprio teste (Roboflow, Angelina, sintético e páginas montadas). São reportadas precisão, revocação, Hmean, mAP@0,5 e mAP@0,5:0,95, além de métricas ponta a ponta em que a célula só conta como acerto quando a caixa coincide (IoU de pelo menos 0,5) e o padrão de pontos está correto, e a taxa de erro de caracteres (CER) da sequência lida. A velocidade é medida em CPU. No teste em português, a leitura de cada foto é comparada ao texto digitado, com CER, proporção de leituras exatas e proporção de fotos em que o texto aparece na leitura.

Resultado preliminar do método clássico, obtido no dataset de referência: Hmean de 0,95 nas imagens sintéticas e zero nas fotos reais do Roboflow e do Angelina, o que confirma na prática a limitação dos métodos de reconstrução de grade descrita na literatura (OVODOV, 2021).
