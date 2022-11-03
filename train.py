from datasets import load_dataset #, Features, Value, Dataset
# import pandas as pd
# from tqdm import tqdm
# from pandarallel import pandarallel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import numpy as np
import evaluate
from nltk.tokenize import sent_tokenize
import os



class Data_Preprocessor():
    def __init__(self, m_cp = 'google/mt5-small'):
        self.max_input_length = 512
        self.max_target_length = 30
        self.plm = m_cp
        self.tokenizer = AutoTokenizer.from_pretrained(self.plm)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.plm)


    def custom_load_data(self):
        train_set = load_dataset('csv', data_files='train_df.csv') ### 3210/11943
        test_set = load_dataset('csv', data_files='test_df.csv') ### 100/1491
        val_set = load_dataset('csv', data_files='val_df.csv') ### 100/1491
        test_set['test'] = test_set.pop('train')
        val_set['validation'] = val_set.pop('train')
        return train_set, test_set, val_set

    def show_samples(self, dataset, num_samples=3, seed=42):
        sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
        for example in sample:
            print(f"\n'>> News: {example['text_source']}'")
            print(f"'>> Abstract: {example['summary_target']}'")

    def _preprocess_func(self,sample):
        model_inputs = self.tokenizer(
            sample['text_target'],
            max_length = self.max_input_length,
            truncation = True, 
            padding = True
        )
        labels = self.tokenizer(
            sample['summary_target'],
            max_length = self.max_target_length,
            truncation = True,
            padding = True
        )
        model_inputs['labels'] = labels['input_ids']
        model_inputs['labels_mask'] = labels['attention_mask']
        return model_inputs
    
    def run(self):
        train_set, test_set, val_set = self.custom_load_data()
        tokenized_train_set = train_set.map(self._preprocess_func, batched=True)
        tokenized_test_set = test_set.map(self._preprocess_func, batched=True)
        tokenized_val_set = val_set.map(self._preprocess_func, batched=True)


        tokenized_train_set = tokenized_train_set.remove_columns(
            train_set['train'].column_names
        )
        tokenized_test_set = tokenized_test_set.remove_columns(
            test_set['test'].column_names
        )
        tokenized_val_set = tokenized_val_set.remove_columns(
            val_set['validation'].column_names
        )

        return tokenized_train_set, tokenized_test_set, tokenized_val_set

class XLS_Trainer():
    def __init__(self, batch_size = 8, num_train_epochs = 8, m_cp = 'google/mt5-small'):
        self.rouge_score = evaluate.load("rouge")
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.plm = m_cp
        self.tokenizer = AutoTokenizer.from_pretrained(self.plm)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.plm)
        # logging_steps = len(tokenized_train_set['train']) // batch_size
        model_name = self.plm.split('/')[-1]
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model = self.model)
        self.args = Seq2SeqTrainingArguments(
            output_dir=f'{model_name}-ft-en-chs',
            evaluation_strategy='epoch',
            learning_rate = 5.6e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay = 0.01,
            save_total_limit=3,
            num_train_epochs=self.num_train_epochs,
            predict_with_generate=True,
            logging_steps = 400,
            push_to_hub = False
        )
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = self.rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    def run_train(self, tokenized_train_set, tokenized_val_set):
        trainer = Seq2SeqTrainer(
            self.model,
            self.args,
            train_dataset=tokenized_train_set["train"],
            eval_dataset=tokenized_val_set["validation"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    data_preprocessor = Data_Preprocessor()
    tkn_train_set, tkn_test_set, tkn_val_set = data_preprocessor.run()
    xls_trainer = XLS_Trainer()
    xls_trainer.run_train(tkn_train_set,tkn_val_set)

if __name__ == '__main__':
    main()