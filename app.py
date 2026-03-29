from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit, disconnect
import os, requests as req, zipfile, io, json, struct, threading
from vosk import Model, KaldiRecognizer
import queue
import time
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", engineio_logger=False, 
                   ping_timeout=3600, ping_interval=60, max_http_buffer_size=1e7,
                   async_mode='threading')

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip"
MODEL_DIR = "model"
model = None
sessions = {}
session_lock = threading.Lock()

# Configurações de filtro
MIN_WORD_LENGTH = 2
SILENCE_THRESHOLD = 0.005
NOISE_GATES = ['ah', 'eh', 'uhm', 'hum', 'hmm', 'uh', 'mm', 'er', 'um', 'ã', 'é']

class AudioSession:
    def __init__(self, sid, model):
        self.sid = sid
        self.rec = KaldiRecognizer(model, 16000)
        self.queue = queue.Queue(maxsize=100)
        self.running = True
        self.last_final = ""
        self.silence_frames = 0
        self.worker = threading.Thread(target=self._process, daemon=True)
        self.worker.start()
    
    def estimate_noise_level(self, pcm_data):
        """Estima nível de ruído do áudio"""
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.sqrt(np.mean(audio ** 2))
        return energy
    
    def is_silence(self, energy):
        """Detecta se é silêncio"""
        return energy < SILENCE_THRESHOLD
    
    def filter_words(self, words):
        """Filtra palavras de baixa qualidade"""
        filtered = []
        for word in words:
            word = word.lower().strip()
            
            if not word:
                continue
            if len(word) < MIN_WORD_LENGTH:
                continue
            if word in NOISE_GATES:
                continue
            if word.isdigit() and len(word) == 1:
                continue
            if all(c in 'aeiouáéíóú' for c in word) and len(word) < 3:
                continue
            
            filtered.append(word)
        
        return filtered
    
    def _process(self):
        consecutive_silence = 0
        
        while self.running:
            try:
                data = self.queue.get(timeout=1)
                if data is None:
                    break
                
                energy = self.estimate_noise_level(data)
                
                if self.is_silence(energy):
                    consecutive_silence += 1
                else:
                    consecutive_silence = 0
                
                if consecutive_silence > 5:
                    self.rec.Reset()
                    consecutive_silence = 0
                    continue
                
                if self.rec.AcceptWaveform(data):
                    try:
                        result = json.loads(self.rec.Result())
                        words = result.get('result', [])
                        filtered_words = self.filter_words(words)
                        
                        if filtered_words and len(filtered_words) >= 2:
                            text = ' '.join(filtered_words)
                            self.last_final = text
                            print(f"[FINAL] {text} | Energy: {energy:.4f}")
                            socketio.emit('final', {
                                'text': text,
                                'words': filtered_words,
                                'confidence': float(energy)
                            }, room=self.sid)
                        else:
                            print(f"[DESCARTADO] Poucos palavras válidas: {words} | Energy: {energy:.4f}")
                    except Exception as e:
                        print(f"[ERRO FINAL] {e}")
                else:
                    try:
                        partial = json.loads(self.rec.PartialResult())
                        if partial.get('partial'):
                            partial_text = partial['partial']
                            words = partial_text.split()
                            filtered_words = self.filter_words(words)
                            
                            if filtered_words and energy > SILENCE_THRESHOLD:
                                text = ' '.join(filtered_words)
                                print(f"[PARCIAL] {text} | Energy: {energy:.4f}")
                                socketio.emit('partial', {
                                    'text': text,
                                    'words': filtered_words,
                                    'confidence': float(energy)
                                }, room=self.sid)
                    except Exception as e:
                        print(f"[ERRO PARCIAL] {e}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERRO PROCESSAMENTO {self.sid}] {e}")
                break
    
    def add_audio(self, data):
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            print(f"[AVISO] Fila cheia para {self.sid}")
    
    def close(self):
        self.running = False
        self.queue.put(None)
        self.worker.join(timeout=2)

def setup_model():
    global model
    if not os.path.exists(MODEL_DIR):
        print("Baixando modelo (sem timeout)...")
        try:
            response = req.get(MODEL_URL, stream=True)
            if response.status_code != 200:
                raise Exception(f"Erro ao baixar: {response.status_code}")
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024
            buffer = io.BytesIO()
            start_time = time.time()
            
            print(f"Tamanho total: {total_size / (1024*1024):.1f} MB\n")
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    buffer.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed / (1024*1024) if elapsed > 0 else 0
                    percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                    
                    if total_size > 0:
                        remaining = (total_size - downloaded) / (speed * 1024*1024) if speed > 0 else 0
                        print(f"[{'='*int(percent/2)}{' '*int(50-percent/2)}] {percent:5.1f}% | {downloaded//(1024*1024):4d}MB / {total_size//(1024*1024):3d}MB | {speed:6.2f} MB/s | ETA: {int(remaining):3d}s", end='\r')
            
            print("\nExtraindo arquivos...", end='', flush=True)
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as z:
                z.extractall(MODEL_DIR)
            print(" ✓")
        except req.exceptions.Timeout:
            raise Exception("Timeout ao baixar modelo. Tente novamente.")
        except Exception as e:
            raise Exception(f"Erro ao baixar/extrair modelo: {e}")
    
    model_dirs = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    if not model_dirs:
        raise Exception("Nenhum diretório de modelo encontrado!")
    
    model_path = os.path.join(MODEL_DIR, model_dirs[0])
    print(f"Carregando modelo de: {model_path}")
    model = Model(model_path)
    print("✓ Modelo carregado.")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@socketio.on('connect')
def on_connect(auth):
    with session_lock:
        sid = request.sid
        if model is None:
            emit('error', 'Modelo não carregado')
            disconnect()
            return
        
        sessions[sid] = AudioSession(sid, model)
        print(f"✓ Conectado: {sid}")
        emit('ready', True)

@socketio.on('audio')
def handle_audio(data):
    with session_lock:
        sid = request.sid
        session = sessions.get(sid)
    
    if not session:
        return
    
    try:
        if isinstance(data, list):
            pcm = struct.pack('<' + 'h' * len(data), *data)
        else:
            pcm = data
        
        session.add_audio(pcm)
    except Exception as e:
        print(f"[ERRO ÁUDIO {sid}] {e}")

@socketio.on('disconnect')
def on_disconnect():
    with session_lock:
        sid = request.sid
        session = sessions.pop(sid, None)
    
    if session:
        session.close()
        print(f"✓ Desconectado: {sid}")

if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    setup_model()
    socketio.run(app, debug=False, host='127.0.0.1', port=5000, 
                allow_unsafe_werkzeug=True, use_reloader=False)