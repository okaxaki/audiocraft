from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler
import os
import sys
import base64
import json
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import time


start = time.time()
print(f'Initializing...')
# device='mps': Apple Silicon の GPU
model = AudioGen.get_pretrained('facebook/audiogen-medium', device='mps') 
print(f'Initialized: {round(time.time()-start, 2)} sec')

def _bytes_to_base64(bytes):
    return base64.b64encode(bytes).decode()

def generateBadRequest(self, explain=None):
    self.send_error(400, 'Bad Request', explain)

def generate_audio(desc):
    print(f'Description: {desc}')
    start = time.time()
    model.set_generation_params(duration=3)  # [duration] 秒のファイルを生成
    results = model.generate([desc], progress=True)  # 引数として与えられたテキスト全ての音声を生成
    for idx, one_wav in enumerate(results):
        out = audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f'Generated: {out} in {round(time.time()-start, 2)} sec')
        file = open(out, 'rb')
        return file.read()

class RequestHandler(BaseHTTPRequestHandler):
    def generateResponse(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        url = urlparse(self.path)
        queries = parse_qs(url.query)
        if 'text' in queries:
            text = queries['text'][0]
            wav = generate_audio(text)
            self.generateResponse(wav)
        else:
            self.generateBadRequest()

if __name__ == '__main__':
    from http.server import HTTPServer

    port = 8080
    host = '0.0.0.0'
    server = HTTPServer((host, port), RequestHandler)
    print('Starting server on http://%s:%d, use <Ctrl-C> to stop' % (host, port))
    server.serve_forever()