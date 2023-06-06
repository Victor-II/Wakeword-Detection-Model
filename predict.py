import time
import torchaudio
import torch
from neural_net.dataset import AudioTransforms
from neural_net.model import CNNModel
from data_scripts.record_audio import Listener


def predict(model_state_dict, file_path, sample_rate, threshold=0.5):
    model = CNNModel()
    model.load_state_dict(torch.load(model_state_dict))

    signal, _ = torchaudio.load(file_path)
    spectogram = AudioTransforms.mel_spectogram(signal, sample_rate)
    spectogram = AudioTransforms.reshape(spectogram, (64, 64))
    
    out = model(spectogram.unsqueeze(0))

    if out.item() > threshold:
        print(f'\nWake Word Detected [{out.item()}]')
    else:
        print(f'\nWake Word NOT Detected [{out.item()}]')


def main(args):
    try:
        while True:
            listener = Listener(args)
            frames = []
            print('Begin recording...')
            input(f'Press enter to continue. the recoding will be {args.get("seconds")} seconds. press ctrl + c to exit\n')
            time.sleep(0.2) 
            for _ in range(int((listener.sample_rate / listener.chunk) * listener.record_seconds)):
                data = listener.stream.read(listener.chunk, exception_on_overflow = False)
                frames.append(data)

            listener.save_audio(args.get('save_path'), frames)
            predict(model_state_dict=args.get('model_state_dict'),
                    file_path=args.get('save_path'),
                    sample_rate=args.get('sample_rate'),
                    device=args.get('device'),
                    threshold=args.get('threshold')
                    )

    except KeyboardInterrupt:
        print('Keyboard Interrupt')


if __name__ == "__main__":

    args = {
        'sample_rate': 8000,
        'seconds': 2,
        'save_path': 'C:/Users/Victor/Desktop/crdmProiect/temp.wav',
        'model_state_dict': 'C:/Users/Victor/Desktop/crdmProiect/wakeword/saved_models/cnn_model.pt',
        'threshold': 0.5
    }

    main(args)