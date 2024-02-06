import os
import re
from pathlib import Path
from datetime import datetime
from functools import reduce
import pandas as pd
import torch
from torch import Tensor, nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers.wandb import WandbLogger

data_dir = Path("./data")
base_model = "cl-tohoku/bert-base-japanese-v3"

def process_livedoor_file(f: Path) -> dict[str, dict]:
    """Livedoorニュースコーパスのファイルを読み込んで辞書を返す
    Args:
        f (Path): ファイルパス
    Returns:
        dict[str, dict]:
            {
                "url": url,
                "datetime": datetime,
                "title": title,
                "content": content
            }
    """
    with f.open() as h:
        lines = h.readlines()
    
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = [line for line in lines if line != ""]

    tz_pattern = re.compile(r"(?P<sign>\+|-)(?P<hour>\d{2})(?P<colon>:)?(?P<minute>\d{2})$")
    mo = tz_pattern.search(lines[1])
    assert mo, f"tz_pattern didn't match: {text}"
    if not mo.group("colon"):
        lines[1] = lines[1][:mo.start("minute")] + ":" + lines[1][mo.start("minute"):]

    data = {
        "url": lines[0],
        "datetime": datetime.fromisoformat(lines[1]),
        "title": lines[2],
        "content": "\n".join(lines[3:])
    }
    assert data["url"].startswith("http")
    assert data["content"]
    return data

def concat_livedoor_data(a, x):
    category, fs = x
    l = [{"category": category} | process_livedoor_file(f) for f in fs]
    return a + l


class LivedoorDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    def __init__(self, data: pd.DataFrame, cat2y) -> None:
        print(len(data))
        self.x = self.tokenizer(
            data.content.tolist(),
            max_length=self.tokenizer.model_max_length,
            padding=True, truncation=True, return_tensors='pt'
        )        
        self.y = torch.tensor([cat2y[cat] for cat in data.category])
    
    def __len__(self) -> int:
        return self.y.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (self.x["input_ids"][idx], self.x["attention_mask"][idx]), self.y[idx]

categories = [c.name for c in (data_dir / "text").glob("*") if c.is_dir()]
cat2y = {c: i for i, c in enumerate(categories)}

livedoor_files = {c: list((data_dir / "text" / c).glob(f"{c}-*.txt")) for c in categories}
livedoor_records = reduce(concat_livedoor_data, livedoor_files.items(), list())
livedoor_data = pd.DataFrame.from_records(livedoor_records)
livedoor_data = livedoor_data.sample(frac=1, random_state=20)
len_livedoor_data = len(livedoor_data)
n = len_livedoor_data // 5
livedoor_data["split"] = ["test"] * n + ["valid"] * n + ["train"] * (len_livedoor_data - 2*n)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

data_loader = {
    split: DataLoader(
        LivedoorDataset(livedoor_data[livedoor_data.split == split], cat2y),
        batch_size=32,
        shuffle=(split == "train"),
        num_workers=4,
    ) for split in ["train", "valid", "test"]
}

class LivedoorNewsClassifier(L.LightningModule):
    def __init__(self, num_classes) -> None:
        super(__class__, self).__init__()
        self.save_hyperparameters()
        self.training_step_outputs = list()
        self.validation_step_outputs = list()
        self.num_classes = num_classes
        self.embedding_model = AutoModel.from_pretrained(base_model)
        self.embedding_size = self.embedding_model.pooler.dense.in_features
        self.fc = nn.Linear(self.embedding_size, self.num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)  
        nn.init.normal_(self.fc.bias, 0)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ids, attention_mask = x
        outputs = self.embedding_model(input_ids, attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        x = self.fc(embeddings)
        return x

    def configure_optimizers(self) -> optim.Optimizer:
        torch.set_float32_matmul_precision('high')

        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.embedding_model.encoder.layer[-1].parameters():
            param.requires_grad = True
                
        for param in self.fc.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam([
            {"params": self.fc.parameters(), "lr": 1e-4},
            {"params": self.embedding_model.encoder.layer[-1].parameters(), "lr": 1e-5}
        ])
        return optimizer

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_pred_logit = self(x)
        loss = nn.functional.cross_entropy(y_pred_logit, y)
        correct = (y_pred_logit.argmax(dim=1) == y).type(torch.float)
        self.training_step_outputs.append({"loss": loss, "correct": correct})
        return loss 
    
    def on_train_epoch_end(self) -> None:
        train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        train_accuracy = torch.cat([x["correct"] for x in self.training_step_outputs]).mean()
        self.log_dict({"loss": train_loss, "accuracy": train_accuracy}, prog_bar=True)
        self.training_step_outputs.clear()
        return
    
    def validation_step(self, batch, batch_index) -> torch.Tensor:
        x, y = batch
        y_pred_logit = self(x)
        loss = nn.functional.cross_entropy(y_pred_logit, y)
        correct = (y_pred_logit.argmax(dim=1) == y).type(torch.float)
        self.validation_step_outputs.append({"val_loss": loss, "val_correct": correct})
        return loss

    def on_validation_epoch_end(self) -> None:
        valid_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        valid_accuracy = torch.cat([x["val_correct"] for x in self.validation_step_outputs]).mean()
        self.log_dict({"valid_loss": valid_loss, "valid_accuracy": valid_accuracy}, prog_bar=True)
        self.validation_step_outputs.clear()
        return
    
    def predict_step(self, batch, batch_index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        y_pred_logit = self(x)
        y_pred = y_pred_logit.argmax(dim=1)
        return y_pred.cpu(), y.cpu()

callbacks = list()
callbacks.append(EarlyStopping(monitor="valid_loss", patience=5))

checkpoint_dir = Path("logs")
_ = list(map(lambda x: x.unlink(), checkpoint_dir.glob("*.ckpt")))
checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",
    filename="sample-{epoch:02d}-{valid_loss:.03f}-{valid_accuracy:.03f}",
    save_top_k=3,
    mode="min",
    dirpath=checkpoint_dir
)
callbacks.append(checkpoint_callback)
model = LivedoorNewsClassifier(num_classes=len(categories))
wandb_logger = WandbLogger(project=f"livedoor-news-classification-{base_model.replace('/', '-')}")
trainer = L.Trainer(
        max_epochs=40,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger
)
trainer.fit(
    model,
    train_dataloaders=data_loader["train"],
    val_dataloaders=data_loader["valid"]
)

from sklearn.metrics import classification_report

def replace_labels(cr_dict: dict, category_name: dict) -> dict:
    new_cr_dict = {}
    for key, value in cr_dict.items():
        if key in ["accuracy", "macro avg", "weighted avg"]:
            new_cr_dict[key] = value
        else:
            new_key = category_name[int(float(key))]
            new_cr_dict[new_key] = value
    
    return new_cr_dict

best_model = LivedoorNewsClassifier.load_from_checkpoint(
    checkpoint_path=checkpoint_callback.best_model_path
)
predictions = trainer.predict(best_model, data_loader["test"])
y_pred, y_true = reduce(lambda a, x: (a[0] + x[0].tolist(), a[1] + x[1].tolist()), predictions, (list(), list()))
cat_pred = list(map(lambda x:categories[x], y_pred))
cr_dict = classification_report(
    y_true,
    y_pred,
    output_dict=True
)
new_cr_dict = replace_labels(cr_dict, categories)
df = pd.DataFrame(new_cr_dict).T
df.to_markdown(f"classification_report.md")
