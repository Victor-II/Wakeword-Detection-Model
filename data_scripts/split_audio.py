import os
from pydub import AudioSegment
from pydub.utils import make_chunks

def main(args):

    audio = AudioSegment.from_file(args.get('file_path'))
    length = args.get('seconds') * 1000 # this is in miliseconds
    chunks = make_chunks(audio, length)
    names = []
    for i, chunk in enumerate(chunks):
        _name = args.get('file_path').split("/")[-1]
        name = "{}_{}".format(i, _name)
        wav_path = os.path.join(args.get('save_path'), name)
        chunk.export(wav_path, format="wav")
    return names


if __name__ == "__main__":

    args = {
        'seconds': 3,
        'file_path': 'File/Path',
        'save_path': 'Save/Path'
    }

    main(args)
