import pandas as pd
import torch.utils.data


# region Functions

def convert_to_prompt(row):
    schema = row['schema']
    question = row['question']

    tables_and_relations = [i for i in schema.split("|") if i.strip() != ""]
    tables = [a for a in tables_and_relations if ":" in a]
    relations = [a for a in tables_and_relations if ":" not in a]

    tables = {t.split(":")[0].strip(): [a.strip() for a in t.split(":")[1].split(",")] for t in tables}
    tables_str = "\n".join([f"Table Name: {k} and columns: {', '.join(v)}" for k, v in tables.items()])

    relations_str = []
    for relation in relations:
        s1, s2 = relation.split("=")
        t1, col1 = s1.split(".")
        t2, col2 = s2.split(".")
        s = f"Column {col1.strip()} of table {t1.strip()} can be joined on {col2.strip()} of table {t2.strip()}"
        relations_str.append(s)
    relations_str = "\n".join(relations_str)

    prompt = f"""
    Given an input question, generate a sql query by choosing one or multiple of the following tables

    For this problem you can use the following tables and their schema

    {tables_str}

    Relations between tables: 

    {relations_str}

    Question: {question}
    """

    return prompt


def convert_to_statement(row):
    statement = f"translate English to SQL: {row['question']} given table's name with schema: {row['schema']}"
    return statement

# endregion Functions


class EnglishSQLDataset(torch.utils.data.Dataset):
    # region Constructor

    def __init__(self, csv_file, tokenizer, max_source_length=512, max_target_length=512, preprocessing_type='prompt'):
        if isinstance(csv_file, str):
            self.data: pd.DataFrame = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.data = data = csv_file
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        if preprocessing_type == 'prompt':
            self.data['input_to_model'] = self.data.apply(convert_to_prompt, axis=1)
        else:
            self.data['input_to_model'] = self.data.apply(convert_to_statement, axis=1)

    # endregion Constructor

    # region Functions

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        encoding = self.tokenizer([item['input_to_model']], padding="max_length", max_length=self.max_source_length,
                                  truncation=True, return_tensors="pt")
        target_encoding = self.tokenizer([item['sql']], padding="max_length", max_length=self.max_target_length,
                                         truncation=True, return_tensors="pt")
        labels = target_encoding['input_ids'][0]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encoding['input_ids'][0],
            "attention_mask": encoding['attention_mask'][0],
            "labels": labels
        }

    def __len__(self):
        return len(self.data)

    # endregion Functions
