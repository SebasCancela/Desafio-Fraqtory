# Desafio-Fraqtory

1. `frames_extraction.py`

Este script extrai um número fixo de `n` frames distribuídos uniformemente ao longo de cada vídeo `.mp4` da pasta `videos`. Os frames são guardados em subpastas dentro de `frames_extracted`, com o nome correspondente a cada vídeo.

2. `frames_undistorition.py`

Corrige a distorção das imagens extraídas, assumindo que foram captadas com uma lente tipo olho-de-peixe (fisheye). Cada imagem é:
- Recortada 8 píxeis na parte superior para igualar a proporção da escala de uma imagem 16x9.
- Corrigida usando parâmetros intrínsecos e distorção da câmara.
- Redimensionada para `1280x720` e guardada nas pastas `frames_extracted/undistorted` e `model/data/images`.

3. `frames_anotation.py`

Este script copia as anotações base de um ficheiro original (`points_positions.json`, que apenas contém as anotações dos pontos de 1 frame de cada vídeo) e aplica-as a todas as imagens que pertençam ao mesmo vídeo, guardando as anotações no ficheiro `points_positions.json`, e criando as imagens com os pontos marcados, guardadas em `frames_extracted/annotated`. 

4. `convert_data.py`

Converte as anotações exportadas do Label Studio (no formato JSON) para o formato esperado pelo modelo. A conversão:
- Transforma as coordenadas percentuais em coordenadas absolutas (píxeis em 1280x720 de resolução).
- Reorganiza os pontos numa ordem específica.
- Divide os dados entre treino e validação.
- Guarda dois ficheiros: `data_train.json` e `data_val.json`.

5. `augment_data.py`

Realiza a augmentation dos dados. Para cada imagem, é escolhida aleatoriamente uma de entre várias transformações:
- Inversão vertical
- Inversão vertical e horizontal
- Mudança de brilho, contraste e tonalidade (hue)
- Inversão horizontal e mudança de tonalidade (hue)

Os novos frames são guardados em `model/augmented_data/images`, e os dados são guardados como `data_train.json` e `data_val.json`.

6. `model`

Pasta com o modelo utilizado, `https://github.com/yastrebksv/TennisCourtDetector`, e onde também foram criadas as pastas `data`, `augmented_data`, `trained_models` (com os 2 modelos que foram treinados) e `results` (para se verificar o resultado do modelo quando se escolhe uma imagem para prever os pontos)
