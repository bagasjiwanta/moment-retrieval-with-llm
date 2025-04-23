# https://github.com/jayleicn/moment_detr?tab=readme-ov-file

from lightning import LightningDataModule

class QVHighlightsDataset(LightningDataModule):
    def __init__(self):
        super().__init__()
    
    def setup(self, stage):
        return super().setup(stage)
    
    def prepare_data(self):
        return super().prepare_data()
    
    def train_dataloader(self):
        return super().train_dataloader()
    
    def val_dataloader(self):
        return super().val_dataloader()
    
    def test_dataloader(self):
        return super().test_dataloader()