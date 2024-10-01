# Original Code: https://blog.peddals.com/apple-mps-to-generate-audio-with-meta-audiogen/#_MPS

from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import argparse
import time

start = time.time()
model = AudioGen.get_pretrained('facebook/audiogen-medium', device='mps') # device='mps': Apple Silicon の GPU
model.set_generation_params(duration=3)  # [duration] 秒のファイルを生成
print(f'初期化: {round(time.time()-start, 2)} 秒')

start = time.time()
def generate_audio(descriptions):
  wav = model.generate(descriptions, progress=True)  # 引数として与えられたテキスト全ての音声を生成
  
  for idx, one_wav in enumerate(wav):
      # {idx}.wav というファイルを生成。音の大きさ loudness は -14 db LUFS で平準化
      audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
      print(f'{idx}.wav を生成')
      print(f'生成時間: {round(time.time()-start, 2)} 秒')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio based on descriptions.")
    parser.add_argument("descriptions", nargs='+', help="List of descriptions for audio generation")
    args = parser.parse_args()
    
    generate_audio(args.descriptions)