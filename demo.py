from pathlib import Path

from data_tokenize import TranslateDataset

train_dataset = TranslateDataset(
    data_name=Path("./data/raw_data.csv"), dateset_type="train", use_cache=True
)
