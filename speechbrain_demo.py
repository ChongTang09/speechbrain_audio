from speechbrain.pretrained import EncoderDecoderASR
import speechbrain as sb

import audiofile as af
import sounddevice as sd

asr_model = EncoderDecoderASR.from_hparams(
    source='speechbrain/asr-wav2vec2-commonvoice-en',
    savedir='pretrained_models/asr-wav2vec2-commonvoice-en',
    run_opts={"device": 'cuda'}
)

def record_audio(filename, seconds):
    fs = 44100
    print("recording {} ({}s) ...".format(filename, seconds))
    y = sd.rec(int(seconds*fs), samplerate=fs, channels=2)
    sd.wait()
    y = y.T
    af.write(filename, y, fs)
    print(" ... saved to {}".format(filename))

def main():
    while True:
        record_audio('temp.wav', 10)
        print(asr_model.transcribe_file('temp.wav'))

if __name__ == "__main__":
    main()

# import wave
# from dataclasses import dataclass, asdict

# import pyaudio

# @dataclass
# class StreamParams:
#     format: int = pyaudio.paInt16
#     channels: int = 1
#     rate: int = 44100
#     frames_per_buffer: int = 1024
#     input: bool = True
#     output: bool = False
#     input_device_index: int = 1 

#     def to_dict(self) -> dict:
#         return asdict(self)


# class Recorder:
#     """Recorder uses the blocking I/O facility from pyaudio to record sound
#     from mic.

#     Attributes:
#         - stream_params: StreamParams object with values for pyaudio Stream
#             object
#     """
#     def __init__(self, stream_params: StreamParams) -> None:
#         self.stream_params = stream_params
#         self._pyaudio = None
#         self._stream = None
#         self._wav_file = None

#     def record(self, duration: int, save_path: str) -> None:
#         """Record sound from mic for a given amount of seconds.

#         :param duration: Number of seconds we want to record for
#         :param save_path: Where to store recording
#         """
#         print("Start recording...")
#         self._create_recording_resources(save_path)
#         self._write_wav_file_reading_from_stream(duration)
#         self._close_recording_resources()
#         print("Stop recording")

#     def _create_recording_resources(self, save_path: str) -> None:
#         self._pyaudio = pyaudio.PyAudio()
#         self._stream = self._pyaudio.open(**self.stream_params.to_dict())
#         self._create_wav_file(save_path)

#     def _create_wav_file(self, save_path: str):
#         self._wav_file = wave.open(save_path, "wb")
#         self._wav_file.setnchannels(self.stream_params.channels)
#         self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
#         self._wav_file.setframerate(self.stream_params.rate)

#     def _write_wav_file_reading_from_stream(self, duration: int) -> None:
#         for _ in range(int(self.stream_params.rate * duration / self.stream_params.frames_per_buffer)):
#             audio_data = self._stream.read(self.stream_params.frames_per_buffer)
#             self._wav_file.writeframes(audio_data)

#     def _close_recording_resources(self) -> None:
#         self._wav_file.close()
#         self._stream.close()
#         self._pyaudio.terminate()


# if __name__ == "__main__":
#     asr_model = EncoderDecoderASR.from_hparams(
#     source='speechbrain/asr-wav2vec2-commonvoice-en',
#     savedir='pretrained_models/asr-wav2vec2-commonvoice-en'
#     )

#     stream_params = StreamParams()
#     recorder = Recorder(stream_params)
#     while True:
#         recorder.record(10, "temp.wav")
#         print(asr_model.transcribe_file('temp.wav'))
