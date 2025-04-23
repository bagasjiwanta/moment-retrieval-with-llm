# https://prior.allenai.org/projects/charades
# https://github.com/jiyanggao/TALL?tab=readme-ov-file
Link_Original_55Gb = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip"
Link_480p_13Gb = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"

from lightning import LightningDataModule
from project.utils import download_large_file
import shutil

class CharadesSTADataset(LightningDataModule):
    def __init__(
        self,
        base_dir="",
        use_480p=False,
        download_dir="",
        download_filename="charades.zip"
    ):
        self.download_link = Link_Original_55Gb
        if use_480p:
            self.download_link = Link_480p_13Gb
        self.base_dir = base_dir
        self.download_dir = download_dir
        self.filename = download_filename
        self.save_hyperparameters()
        super().__init__()

    def prepare_data(self):
        download_large_file(self.download_link, self.download_dir + self.filename)
        shutil.unpack_archive()

    def setup(self, stage):
        return super().setup(stage)

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
