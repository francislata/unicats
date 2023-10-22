from glob import glob
from tqdm import tqdm

import argparse
import librosa
import os
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(
        prog="LibriTTS sample rate converter",
        usage="This is used to convert audio files to a specific sampling rate"
    )

    parser.add_argument(
        "audio_path",
        help="the path to the audio that is a glob path",
        type=str
    )

    parser.add_argument(
        "dest_path",
        help="the path of where to save the audio",
        type=str
    )

    parser.add_argument(
        "sample_rate",
        help="the sample rate to convert the audio to",
        type=int
    )

    args = parser.parse_args()
    audio_paths = glob(args.audio_path, recursive=True)

    for audio_path in tqdm(audio_paths, desc="Resampling audio"):
        filename = os.path.basename(audio_path)
        dest_path = args.dest_path

        for folder in os.path.splitext(filename)[0].split("_")[:2]:
            dest_path = os.path.join(dest_path, folder)
            
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        au, sr = librosa.load(audio_path, sr=args.sample_rate)
        sf.write(os.path.join(dest_path, filename), au, sr)

if __name__ == "__main__":
    main()