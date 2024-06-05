import pytorch_lightning as pl
import torchmetrics
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import ASTForAudioClassification, ASTConfig
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from utils.paths import paths


class ASTLightningModel(pl.LightningModule):
    def __init__(self, num_classes, id2label, label2id, learning_rate=1e-5):
        super(ASTLightningModel, self).__init__()
        self.save_hyperparameters()
        pretrained_model_path = paths.Pretrained / "audioset_10_10_0.4593.pth"

        new_config = ASTConfig(
            architectures=["ASTForAudioClassification"],
            attention_probs_dropout_prob=0.1,
            frequency_stride=10,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            hidden_size=768,
            layer_norm_eps=1e-12,
            max_length=1024,
            model_type="audio-spectrogram-transformer",
            num_attention_heads=12,
            num_hidden_layers=12,
            num_mel_bins=128,
            patch_size=16,
            qkv_bias=True,
            time_stride=10,
            torch_dtype="float32",
            transformers_version="4.25.0.dev0",
            num_labels=num_classes
        )
        new_config.id2label = id2label
        new_config.label2id = label2id

        self.model = ASTForAudioClassification(new_config)

        if not (pretrained_model_path).exists():
            paths.download_pretrained()

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sd = torch.load(pretrained_model_path, map_location=self._device)
        self.model.load_state_dict(sd, strict=False)

        for layer in self.model.audio_spectrogram_transformer.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.model.audio_spectrogram_transformer.encoder.layer[6:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
        self.model = torch.nn.DataParallel(self.model).to(self._device)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.train_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
        self.val_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
        self.test_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if len(batch) == 0:
            return None
        inputs, labels = batch
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        output = self(inputs)
        logits = output.logits
        loss = self.criterion(logits, labels)

        # Calculate and store metrics
        self.train_metrics = {
            "train_loss": loss,
            "train_acc": self.train_accuracy(logits, labels),
            "train_f1": self.train_f1(logits, labels),
            "train_precision": self.train_precision(logits, labels),
            "train_recall": self.train_recall(logits, labels),
            "train_auc": self.train_auc(logits, labels)
        }

        for key, value in self.train_metrics.items():
            self.log(key, value, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 0:
            return None
        inputs, labels = batch
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        output = self(inputs)
        logits = output.logits
        loss = self.criterion(logits, labels)

        # Calculate and store metrics
        self.val_metrics = {
            "val_loss": loss,
            "val_acc": self.val_accuracy(logits, labels),
            "val_f1": self.val_f1(logits, labels),
            "val_precision": self.val_precision(logits, labels),
            "val_recall": self.val_recall(logits, labels),
            "val_auc": self.val_auc(logits, labels)
        }

        for key, value in self.val_metrics.items():
            self.log(key, value, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        if len(batch) == 0:
            return None
        inputs, labels = batch
        inputs, labels = inputs.to(self._device), labels.to(self._device)
        output = self(inputs)
        logits = output.logits
        loss = self.criterion(logits, labels)

        # Calculate and store metrics
        self.test_metrics = {
            "test_loss": loss,
            "test_acc": self.test_accuracy(logits, labels),
            "test_f1": self.test_f1(logits, labels),
            "test_precision": self.test_precision(logits, labels),
            "test_recall": self.test_recall(logits, labels),
            "test_auc": self.test_auc(logits, labels)
        }

        for key, value in self.test_metrics.items():
            self.log(key, value)

        return loss

    def on_train_epoch_end(self):
        # Print training metrics
        print("\nEpoch Training Metrics:")
        for key, value in self.train_metrics.items():
            print(f"{key}: {value}")

    def on_validation_epoch_end(self):
        # Print validation metrics
        print("\nEpoch Validation Metrics:")
        for key, value in self.val_metrics.items():
            print(f"{key}: {value}")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }