import argparse
from data.dataloader import get_dataloaders 
from models.mocov2 import MocoV2Model
from models.classifier import Classifier
from utils.helpers import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    parser = argparse.ArgumentParser(description='MoCo v2')
    parser.add_argument('--path_to_train', type=str, required=True)
    parser.add_argument('--path_to_test', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--use_toy_dataset', action='store_true', help='Use a toy dataset for quick tests')
    parser.add_argument('--checkpoint_path', type=str, required=False, help='Path to the saved model checkpoint')
    args = parser.parse_args()

    seed_everything(args.seed)

    dataloader_train_moco, dataloader_train_classifier, dataloader_test = get_dataloaders(
        args.path_to_train, args.path_to_test, args.batch_size, args.num_workers, use_toy_dataset=args.use_toy_dataset)
    
    logger = TensorBoardLogger("tb_logs", name="mocov2")

    # model = MocoV2Model(max_epochs=args.max_epochs)
    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger)
    # trainer.fit(model, dataloader_train_moco)

    model = MocoV2Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path)

    # moco_backbone_state_dict = model.backbone.state_dict()

    # model.eval()
    # classifier = Classifier(model.backbone)
    # classifier.backbone.load_state_dict(moco_backbone_state_dict)
    # trainer = pl.Trainer(max_epochs=args.max_epochs)
    # trainer.fit(classifier, dataloader_train_classifier, dataloader_test)

    classifier_checkpoint_path = './lightning_logs/version_0/checkpoints/epoch=19-step=780.ckpt'
    classifier = Classifier.load_from_checkpoint(checkpoint_path=classifier_checkpoint_path,backbone=model.backbone)

    test_results = trainer.test(classifier, dataloader_test)

    print(test_results)

if __name__ == "__main__":
    main()