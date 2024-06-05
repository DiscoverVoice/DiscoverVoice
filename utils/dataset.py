import numpy as np
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import AmplitudeToDB, FrequencyMasking, TimeMasking
from pydub import AudioSegment, silence
from transformers import ASTFeatureExtractor
from pathlib import Path
import random
import shutil
from collections import defaultdict

# Assuming paths is a module with necessary paths defined
from utils.paths import paths

def save_preprocessed_data(data, file_path):
    torch.save(data, file_path)

def load_preprocessed_data(file_path):
    return torch.load(file_path)

def get_audio_list(audio_dir, recursive=True):
    audio_path = Path(audio_dir)
    audio_files = []

    def recur(current_path):
        for entry in current_path.iterdir():
            if entry.is_dir():
                if recursive:
                    recur(entry)
            elif entry.suffix.lower() in [".wav", ".mp3"]:
                audio_files.append(entry)

    recur(audio_path)
    return audio_files

def create_label_mappings(labels):
    id_to_label = {}
    label_to_id = {}

    for idx, label in enumerate(sorted(labels)):
        id_to_label[str(idx)] = label
        label_to_id[label] = idx
    return id_to_label, label_to_id

def split_datasets(audio_files, train_ratio=0.8, valid_ratio=0.3):
    class_to_files = defaultdict(list)
    for audio_file in audio_files:
        artist_title = audio_file.stem.split(" - ", 1)
        if len(artist_title) == 2:
            artist, title = artist_title
            class_to_files[artist].append(audio_file)

    train_files = []
    valid_files = []
    test_files = []

    remaining_files = []
    for artist, files in class_to_files.items():
        random.shuffle(files)
        train_files.append(files.pop())
        remaining_files.extend(files)

    random.shuffle(remaining_files)
    num_remaining = len(remaining_files)
    num_train = int(num_remaining * train_ratio)
    train_files.extend(remaining_files[:num_train])
    test_files.extend(remaining_files[num_train:])

    num_train_files = len(train_files)
    num_valid = int(num_train_files * valid_ratio)
    valid_files.extend(train_files[:num_valid])
    train_files = train_files[num_valid:]

    return train_files, valid_files, test_files

def copy_files(file_list, destination_dir):
    destination_path = Path(destination_dir)
    destination_path.mkdir(parents=True, exist_ok=True)
    for file in file_list:
        destination_file = destination_path / file.name
        if destination_file.exists():
            print(f"{file.name} overwrite")
        shutil.copy(file, destination_file)

def create_dataset_json(audio_dir, output_json, train_dir, valid_dir, test_dir, recursive=True):
    audio_files = get_audio_list(paths.Datasets / audio_dir, recursive)
    train_files, valid_files, test_files = split_datasets(audio_files)

    def files_to_data_list(files):
        data_list = []
        labels_set = set()

        for audio_file in files:
            artist_title = audio_file.stem.split(" - ", 1)
            if len(artist_title) == 2:
                artist, title = artist_title
                labels_set.add(artist)
                data_list.append({"wav": str(audio_file.name), "labels": artist})
        return data_list, labels_set

    train_data, train_labels = files_to_data_list(train_files)
    valid_data, valid_labels = files_to_data_list(valid_files)
    test_data, test_labels = files_to_data_list(test_files)

    all_labels = train_labels.union(valid_labels).union(test_labels)
    id_to_label, label_to_id = create_label_mappings(all_labels)

    dataset_json = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
        "id2label": id_to_label,
        "label2id": label_to_id,
    }

    with open(paths.json_files / output_json, "w", encoding="utf-8") as json_file:
        json.dump(dataset_json, json_file, ensure_ascii=False, indent=4)

    print(f"Dataset JSON file created at {output_json}")

    copy_files(train_files, paths.Datasets / train_dir)
    copy_files(valid_files, paths.Datasets / valid_dir)
    copy_files(test_files, paths.Datasets / test_dir)

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, feature_extractor, device='cuda:0'):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        :param device: 'cpu' or 'cuda'
        """
        self.datapath = dataset_json_file
        self.device = device
        self.feature_extractor = feature_extractor
        self.preprocessed_dir = paths.Datasets / 'prep' / audio_conf.get("mode")

        with open(dataset_json_file, "r") as fp:
            data_json = json.load(fp)

        if audio_conf.get("mode") == "train":
            self.data = data_json["train"]
        elif audio_conf.get("mode") == "valid":
            self.data = data_json["valid"]
        elif audio_conf.get("mode") == "test":
            self.data = data_json["test"]
        else:
            raise ValueError("Unpermitted mode has been selected. mode->'train'/'valid'/'test' ")

        self.id_to_label = data_json["id2label"]
        self.label_to_id = data_json["label2id"]
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm")
        self.timem = self.audio_conf.get("timem")
        self.mixup = self.audio_conf.get("mixup")
        self.dataset = self.audio_conf.get("dataset")
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        self.skip_norm = self.audio_conf.get("skip_norm") if self.audio_conf.get("skip_norm") else False
        if self.skip_norm:
            print("now skip normalization (use it ONLY when you are computing the normalization stats).")
        else:
            print("use dataset mean {:.3f} and std {:.3f} to normalize the input.".format(self.norm_mean, self.norm_std))
        self.noise = self.audio_conf.get("noise", False)
        self.label_num = len(self.id_to_label)

    def __getitem__(self, index):
        datum = self.data[index]
        target_length = self.audio_conf['target_length']
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_file_path = self.preprocessed_dir / f"{datum['wav']}.pt"

        if preprocessed_file_path.exists():
            inputs = load_preprocessed_data(preprocessed_file_path)
        else:
            audio_path = paths.Datasets / f"{self.audio_conf.get('mode')}" / datum["wav"]
            audio = AudioSegment.from_file(audio_path)

            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
                
            # non_silent_audio = silence.detect_nonsilent(audio, min_silence_len=1000, silence_thresh=-40)
            # non_silent_audio = [audio[start:end] for start, end in non_silent_audio]
            # audio = sum(non_silent_audio)
                
            waveform = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)

            if waveform.shape[1] < target_length:
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]  # (1, max_length)


            inputs = self.feature_extractor(waveform.squeeze(0).numpy(), sampling_rate=audio.frame_rate, return_tensors="pt")
            label = self.label_to_id[datum["labels"]]
            inputs['labels'] = torch.tensor(label).long()

            if self.preprocessed_dir:
                save_preprocessed_data(inputs, preprocessed_file_path)

        input_values = inputs['input_values'].squeeze(0)  # Adjust squeeze if necessary
        labels = inputs['labels']
        return input_values, labels

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    input_values = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return input_values, labels