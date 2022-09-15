from speechbrain.pretrained import EncoderDecoderASR
import speechbrain as sb

import audiofile as af
import sounddevice as sd

from ntp import RequestTimefromNtp

asr_model = EncoderDecoderASR.from_hparams(
    source='speechbrain/asr-wav2vec2-commonvoice-en',
    savedir='pretrained_models/asr-wav2vec2-commonvoice-en',
    run_opts={"device": 'cuda'}
)

def record_audio(filename, seconds):
    fs = 44100
    print("recording {} ({}s) at {}".format(filename, seconds, RequestTimefromNtp()[0]))
    y = sd.rec(int(seconds*fs), samplerate=fs, channels=2)
    sd.wait()
    y = y.T
    af.write(filename, y, fs)
    print(" ... saved to {} at {}".format(filename, RequestTimefromNtp()[0]))

def main():
    while True:
        record_audio('temp.wav', 5)
        print(asr_model.transcribe_file('temp.wav'))

if __name__ == "__main__":
    main()