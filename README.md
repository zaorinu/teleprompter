# teleprompter

Este projeto implementa um mecanismo de reconhecimento de fala usando Vosk no servidor Python, com um cliente web para gravação e exibição do texto.

## Instalação

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute o servidor:
   ```bash
   python app.py
   ```

3. Abra o navegador em `http://localhost:5000` para acessar o site.

## Funcionamento

- O modelo Vosk (português full-band) é baixado automaticamente na primeira execução (~2GB, pode demorar).
- Clique em "Falar" para iniciar o reconhecimento em tempo real via microfone.
- As palavras são exibidas na tela e logadas no console do DevTools palavra por palavra.
- Clique em "Parar" para finalizar e obter o texto final.

## Modelo

Modelo usado: `vosk-model-pt-fb-v0.1.1-20220516_2113` (português, ~2GB RAM).

Se o download falhar (erro 401), baixe manualmente de https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip, extraia em `model/vosk-model-pt-fb-v0.1.1-20220516_2113` e reinicie.

## Notas

- Certifique-se de que o navegador tem permissão para acessar o microfone.
- O áudio é enviado como WAV para o servidor.