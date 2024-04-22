import pytorch_lightning as pl

def seed_everything(seed):
    pl.seed_everything(seed)