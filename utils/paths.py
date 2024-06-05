import os
from pathlib import Path
import sys
import pyrootutils
import wget

class Paths:
    def __init__(self):
        current_path = Path(__file__).resolve()
        self.root_dir = pyrootutils.setup_root(current_path.parent.parent, indicator=".project-root", pythonpath=True)
        os.chdir(self.root_dir)
        if str(self.root_dir) not in sys.path:
            sys.path.append(str(self.root_dir))

        self.Datasets = self.root_dir / 'Datasets'
        self.Configs = self.root_dir / 'Configs'
        self.Logs = self.root_dir / 'Logs'
        self.Results = self.root_dir / 'Results'
        self.json_files = self.root_dir / 'json_files'
        self.Pretrained = self.root_dir / 'Pretrained'
        self.make_sure()

    def make_sure(self):
        self.Datasets.mkdir(parents=True, exist_ok=True)
        self.Configs.mkdir(parents=True, exist_ok=True)
        self.Logs.mkdir(parents=True, exist_ok=True)
        self.Results.mkdir(parents=True, exist_ok=True)
        self.json_files.mkdir(parents=True, exist_ok=True)
        self.Pretrained.mkdir(parents=True, exist_ok=True)

    def download_pretrained(self):
        url = 'https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1'
        wget.download(url, out=str(self.Pretrained.resolve()))

    @staticmethod
    def to(name):
        return Path(name).resolve()


paths = Paths()
