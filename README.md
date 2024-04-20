# Text2SqlConverter

Text2SqlConverter is a Python project for converting English text questions to SQL queries using a fine-tuned sequence-to-sequence model. The project includes functionalities for data preprocessing, model training, and inference.

## Project Structure

The project is structured as follows:
- **data**: Contains data preprocessing codes.
- **models**: Contains model training codes.
- **inference**: Contains model inference codes.

## Description
- **Data Preprocessing**: The `english_sql_dataset.py` file in the `src/data` directory contains functions for preprocessing the English-SQL dataset. It includes functions to convert data rows to prompts or statements for the model.
- **Model Training**: The `train.py` file in the `src/models` directory contains codes for training the text-to-SQL conversion model. It loads the data, tokenizes it, trains the model, and saves the best model checkpoint.
- **Model Inference**: The `inference.py` file in the `src/inference` directory contains codes for generating SQL queries from English questions. It loads the trained model and generates SQL queries either for a single question or in batch mode from a dataset.

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/HovhannesAbgaryan/Text2SqlConverter.git
    cd Text2SqlConverter
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Train the model:

    ```bash
    python src/models/train.py
    ```

4. Run inference:

    ```bash
    python src/inference/inference.py
    ```

## Example

For training the model:

```bash
python src/models/train.py
```

For running inference:

```bash
python src/inference/inference.py
```
