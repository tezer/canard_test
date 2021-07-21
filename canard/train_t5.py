import json

import click
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer

from canard.constants import MODEL_NAME, N_EPOCHS, BATCH_SIZE, HISTORY_MAX_TOKEN_LEN, REWRITE_MAX_TOKEN_LEN

tokenizer = T5Tokenizer.from_pretrained("t5-small")
pl.seed_everything(42)


class CanardDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: T5Tokenizer,
                 history_max_token_len: int = HISTORY_MAX_TOKEN_LEN,
                 rewrite_max_token_len: int = REWRITE_MAX_TOKEN_LEN
                 ):
        self.tokenizer = tokenizer
        self.data = data
        self.history_max_token_len = history_max_token_len
        self.rewrite_max_token_len = rewrite_max_token_len

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        history = data_row['History']
        history_encoding = tokenizer(
            history,
            max_length=self.history_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        rewrite_encoding = tokenizer(
            data_row["Rewrite"],
            max_length=self.rewrite_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = rewrite_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            history=history,
            rewrite=data_row["Rewrite"],
            history_input_ids=history_encoding["input_ids"].flatten(),
            history_attention_mask=history_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_masks=rewrite_encoding["attention_mask"].flatten()
        )

    def __len__(self):
        return len(self.data)


class CanardDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 tokenizer: T5Tokenizer,
                 batch_size: int = 8,
                 history_max_token_len: int = HISTORY_MAX_TOKEN_LEN,
                 rewrite_max_token_len: int = REWRITE_MAX_TOKEN_LEN
                 ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.history_max_token_len = history_max_token_len
        self.rewrite_max_token_len = rewrite_max_token_len

    def setup(self, stage=None):
        self.train_dataset = CanardDataset(
            self.train_df,
            self.tokenizer,
            self.history_max_token_len,
            self.rewrite_max_token_len
        )

        self.test_dataset = CanardDataset(
            self.test_df,
            self.tokenizer,
            self.history_max_token_len,
            self.rewrite_max_token_len
        )

        self.val_dataset = CanardDataset(
            self.val_df,
            self.tokenizer,
            self.history_max_token_len,
            self.rewrite_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12
        )


def load_data(input_folder):
    dev_df = load_from_file(input_folder + "/dev.json")
    train_df = load_from_file(input_folder + "/train.json")
    test_df = load_from_file(input_folder + "/test.json")
    return dev_df, train_df, test_df


def load_from_file(path):
    sep = "|||"
    data_dict = json.load(open(path))
    df = pd.DataFrame(data_dict)
    df.dropna(inplace=True)
    df['History'] = df['History'].str.join(sep) + sep + df['Question']
    df.drop(['QuAC_dialog_id', 'Question_no', "Question"], axis=1, inplace=True)
    return df


class CanardModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        return output.loss, output.logits

    def _step(self, batch, batch_idx):
        input_ids = batch["history_input_ids"]
        attention_mask = batch["history_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_masks"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, outputs = self._step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self._step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs = self._step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)


@click.command()
@click.option('--input_folder',
              type=str,
              prompt='Specify the input folder with dev.jsom, train.json and test.json files',
              help='Enter the path to the folder with the data files.')
def train(input_folder: str):
    val_df, train_df, test_df = load_data(input_folder)
    data_module = CanardDataModule(train_df=train_df, test_df=test_df, val_df=val_df, tokenizer=tokenizer,
                                   batch_size=BATCH_SIZE)
    model = CanardModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name="canard")

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=True,
        callbacks=checkpoint_callback,
        max_epochs=N_EPOCHS,
        gpus=1
    )
    trainer.fit(model, data_module)
    trainer.test()
    click.echo("The best model is saved at ", trainer.checkpoint_callback.best_model_path)


if __name__ == '__main__':
    train()
