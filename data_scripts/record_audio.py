import pyaudio
import wave
import time
import os


class Listener:

    def __init__(self, args):
        self.chunk = 1024
        self.FORMAT = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = args.get('sample_rate')
        self.record_seconds = args.get('seconds')

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)


    def save_audio(self, file_name, frames):
        # print(f'saving file to {file_name}')
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        wf = wave.open(file_name, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()


def main(args):
    if os.listdir(args.get('save_path')) == []:
        index = 0
    else:
        index = int(os.listdir(args.get('save_path'))[-1].split('.')[0]) + 1
    try:
        while True:
            listener = Listener(args)
            frames = []
            print('Begin recording....')
            input(f'Press enter to continue. the recoding will be {args.get("seconds")} seconds. press ctrl + c to exit')
            time.sleep(0.2)  
            for _ in range(int((listener.sample_rate / listener.chunk) * listener.record_seconds)):
                data = listener.stream.read(listener.chunk, exception_on_overflow = False)
                frames.append(data)
            save_path = os.path.join(args.get('save_path'), "{}.wav".format(index))
            listener.save_audio(save_path, frames)
            index += 1
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        print(str(e))


if __name__ == "__main__":

    args = {
        'sample_rate': 8000,
        'seconds': 3600,
        'save_path': 'C:/Users/Victor/Desktop/WakeWordData/wakewords/'
    }
    
    main(args)
