# app.py
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
import os, requests as req, zipfile, io, json, threading, queue
import unicodedata, re
import numpy as np
from vosk import Model, KaldiRecognizer

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip"
MODEL_DIR = "model"

model = None
sessions = {}
lock = threading.Lock()

# =========================
# NORMALIZE
# =========================
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return re.sub(r'[^a-z0-9\s]', '', text)

# =========================
# MATCH
# =========================
def similar(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > 2:
        return False
    score = sum(c1 == c2 for c1, c2 in zip(a, b))
    return score / max(len(a), len(b)) > 0.65

# =========================
# TRACKER
# =========================
class Tracker:
    def __init__(self, script):
        self.words = normalize(script).split()
        self.pos = 0

    def advance(self, rec_words):
        max_jump = 0

        for r in rec_words:
            # procura até 10 palavras à frente
            for i in range(self.pos, min(self.pos + 10, len(self.words))):
                if similar(r, self.words[i]):
                    jump = i - self.pos + 1
                    max_jump = max(max_jump, jump)

        if max_jump > 0:
            self.pos += max_jump

# =========================
# SESSION
# =========================
class Session:
    def __init__(self, sid):
        self.sid = sid
        self.q = queue.Queue(maxsize=30)
        self.running = True

        self.tracker = None
        self.rec = None
        self.last_pos = -1

        threading.Thread(target=self.loop, daemon=True).start()

    def set_script(self, script):
        self.tracker = Tracker(script)
        self.rec = KaldiRecognizer(model, 16000)

    def energy(self, pcm):
        if pcm is None:
            return 0
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return np.sqrt(np.mean(audio**2)) / 32768

    def loop(self):
        while self.running:
            try:
                data = self.q.get(timeout=1)
                if data is None:
                    break

                if not self.rec or not self.tracker:
                    continue

                if isinstance(data, list):
                    data = np.array(data, dtype=np.int16).tobytes()

                e = self.energy(data)

                if e < 0.003:
                    continue

                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    words = normalize(result.get("text", "")).split()

                else:
                    partial = json.loads(self.rec.PartialResult())
                    words = normalize(partial.get("partial", "")).split()

                if not words:
                    continue

                old = self.tracker.pos
                self.tracker.advance(words)

                # evita salto absurdo
                if self.tracker.pos - old > 12:
                    self.tracker.pos = old + 3

                # 🔥 envia só se mudou
                if self.tracker.pos != self.last_pos:
                    self.last_pos = self.tracker.pos

                    socketio.emit("update", {
                        "pos": self.tracker.pos
                    }, room=self.sid)

            except queue.Empty:
                continue

    def add(self, data):
        try:
            self.q.put_nowait(data)
        except:
            pass

    def close(self):
        self.running = False
        self.q.put(None)

# =========================
# MODEL
# =========================
def setup_model():
    global model

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_path = None

    for d in os.listdir(MODEL_DIR):
        full = os.path.join(MODEL_DIR, d)
        if os.path.isdir(full) and "vosk" in d:
            model_path = full

    if not model_path:
        print("Baixando modelo...")
        r = req.get(MODEL_URL, stream=True)
        buf = io.BytesIO()

        for chunk in r.iter_content(1024*1024):
            if chunk:
                buf.write(chunk)

        buf.seek(0)
        with zipfile.ZipFile(buf) as z:
            z.extractall(MODEL_DIR)

        for d in os.listdir(MODEL_DIR):
            full = os.path.join(MODEL_DIR, d)
            if os.path.isdir(full) and "vosk" in d:
                model_path = full

    print("Carregando:", model_path)
    model = Model(model_path)
    print("Modelo pronto")

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# =========================
# SOCKET
# =========================
@socketio.on("connect")
def connect(auth):
    sid = request.sid
    with lock:
        sessions[sid] = Session(sid)
    emit("ready", True)

@socketio.on("set_script")
def set_script(data):
    sessions[request.sid].set_script(data["script"])

@socketio.on("audio")
def audio(data):
    sessions[request.sid].add(data)

@socketio.on("disconnect")
def disconnect_user():
    s = sessions.pop(request.sid, None)
    if s:
        s.close()

if __name__ == "__main__":
    setup_model()
    socketio.run(app, host="127.0.0.1", port=5000)