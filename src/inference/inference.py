import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.data.english_sql_dataset import convert_to_prompt, EnglishSQLDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class Inference:
    # region Constructor

    def __init__(self, device, checkpoint):
        self.device = device

        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # endregion Constructor

    # region Functions

    def generate_sql(self, question, schema):
        prompt = convert_to_prompt({"question": question, "schema": schema})
        input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_sql_batch(self, csv_file, out_file, num_workers=8):
        train_dataset = EnglishSQLDataset(csv_file, self.tokenizer, preprocessing_type='prompt')
        sqls = []
        with torch.no_grad():
            for batch in tqdm(DataLoader(train_dataset, batch_size=32, num_workers=num_workers)):
                outputs = self.model.generate(batch['input_ids'].to("cuda"), max_new_tokens=512)
                for i in outputs:
                    sqls.append(self.tokenizer.decode(i, skip_special_tokens=True))

        data = pd.read_csv(csv_file)
        data['pred'] = sqls
        data.to_csv(out_file, index=False)

    # endregion Functions


if __name__ == '__main__':
    inference = Inference(device='cuda', checkpoint="HovhAbg/text2sql")
    dataset_link = "https://github.com/Metricam/Public_data/raw/master/text-to-sql_from_spider.csv"
    inference.generate_sql_batch(dataset_link, out_file="results.csv")
