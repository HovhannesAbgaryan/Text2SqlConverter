import pandas as pd
import torch
from src.data.english_sql_dataset import EnglishSQLDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration


def train(csv_file, checkpoint="google-t5/t5-small", num_epochs=3, save_path="./text2sql", push_to_hub=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data = pd.read_csv(csv_file)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = EnglishSQLDataset(train_data, tokenizer, preprocessing_type='prompt')
    valid_dataset = EnglishSQLDataset(val_data, tokenizer, preprocessing_type='prompt')

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    eval_dataloader = DataLoader(valid_dataset, batch_size=32)

    optimizer = AdamW(model.parameters(), lr=5e-3)

    num_training_steps = num_epochs * len(train_dataloader)

    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for val_batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                v_loss = outputs.loss
                val_loss += v_loss.item() * len(val_batch)
                val_count += len(val_batch)
            if val_loss / val_count < best_loss:
                best_loss = val_loss / val_count
                model.save_pretrained(save_path)
                if push_to_hub:
                    model.push_to_hub("HovhAbg/text2sql")
                    tokenizer.push_to_hub("HovhAbg/text2sql")


if __name__ == '__main__':
    dataset_link = "https://github.com/Metricam/Public_data/raw/master/text-to-sql_from_spider.csv"
    train(dataset_link)
