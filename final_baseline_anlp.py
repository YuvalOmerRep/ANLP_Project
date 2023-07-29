from datasets import load_dataset
from transformers import AutoConfig , AutoTokenizer , AutoModelForMultipleChoice, TrainingArguments, Trainer, DefaultDataCollator
from evaluate import load
import numpy as np

config = AutoConfig.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
accuracy = load("accuracy")

riddleSense_train = load_dataset('riddle_sense', split='train').shuffle()
riddleSense_val = load_dataset('riddle_sense', split='validation')

label_mapping_dict = {'A':0, 'B':1, "C":2, "D":3, "E":4}
num_to_label_dict = {0:'A', 1:"B", 2:"C", 3:"D", 4:"E"}

"""Preprocess function, splits each sample into five samples where before we had a riddle and five possible answers, and now we have 5 samples each with the riddle and one of the answers."""

def preprocess_function(examples):
    question_headers = examples["question"]
    second_sentences = [
        [f"{header}[SEP]{examples['choices'][i]['text'][j]}" for j in range(5)] for i, header in enumerate(question_headers)
    ]

    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(second_sentences, truncation=True)
    return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}

preprocessed_train = riddleSense_train.map(preprocess_function, batched=True)
preprocessed_val = riddleSense_val.map(preprocess_function, batched=True)

"""A costume DataCollator class for our praticular needs"""

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'answerKey'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        new_labels = []
        for label in labels:
          new_labels.append(label_mapping_dict[label])
        batch["labels"] = torch.tensor(new_labels, dtype=torch.int64)
        return batch

training_args = TrainingArguments("riddle_sense_check", save_strategy="no", label_names=['answerKey'])

model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased", config=config)
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=preprocessed_train,
                  eval_dataset=preprocessed_val,
                  tokenizer=tokenizer,
                  compute_metrics=None,
                  data_collator = DataCollatorForMultipleChoice(tokenizer))

trainer.train()

"""Predict on validation set and calculate accuracy"""

predictions = trainer.predict(preprocessed_val)

def mapping_func(n):
  return label_mapping_dict[n]

preds = np.argmax(np.array(predictions.predictions[1]), axis=1)
labels = np.array([i for i in map(mapping_func, preprocessed_val['answerKey'])])

(labels == preds).sum() / len(labels)