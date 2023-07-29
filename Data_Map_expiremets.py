from transformers import DataCollatorWithPadding
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForMultipleChoice, TrainingArguments, Trainer, \
    DefaultDataCollator
from evaluate import load
import numpy as np
import plotly.express as px
import pickle
import pandas as pd

MODEL_NAME = "bert-base-uncased"

config = AutoConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
accuracy = load("accuracy")

label_mapping_dict = {'A': 0, 'B': 1, "C": 2, "D": 3, "E": 4}
num_to_label_dict = {0: 'A', 1: "B", 2: "C", 3: "D", 4: "E"}


def mapping_func(n):
    return label_mapping_dict[n]


def preprocess_function(examples):
    question_headers = examples["question"]
    second_sentences = [
        [f"{header}[SEP]{examples['choices'][i]['text'][j]}" for j in range(5)] for i, header in
        enumerate(question_headers)
    ]

    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(second_sentences, truncation=True)
    return {k: [v[i: i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}


riddleSense_train = load_dataset('riddle_sense', split='train')
riddleSense_train_map = load_dataset('riddle_sense', split='train')
riddleSense_val = load_dataset('riddle_sense', split='validation')

preprocessed_train = riddleSense_train.map(preprocess_function, batched=True)
riddleSense_train_map_pre = riddleSense_train_map.map(preprocess_function, batched=True)
val_pre = riddleSense_val.map(preprocess_function, batched=True)


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


"""Costum Trainer class for creating Data Maps"""


class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, compute_metrics, data_collator):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                         tokenizer=tokenizer, compute_metrics=compute_metrics, data_collator=data_collator)
        self.acc = []
        self.eval_dataset = eval_dataset
        self.labels = torch.tensor([label_mapping_dict[i] for i in self.eval_dataset["answerKey"]])

    def evaluate(self, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluate(self.eval_dataset, ignore_keys, metric_key_prefix)
        predictions = self.predict(self.eval_dataset)
        probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions[1]), dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        true_label_probs.append(probs.gather(1, self.labels.unsqueeze(1)).squeeze())
        correctness.append((pred_labels == self.labels).float())
        accuracy = torch.mean((self.labels == pred_labels).float()).item()
        acc.append(accuracy)
        return output


true_label_probs = []
correctness = []
training_args = TrainingArguments("riddle_sense_check", save_strategy="no", label_names=['answerKey'],
                                  evaluation_strategy="epoch", num_train_epochs=3)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, config=config)
trainer = CustomTrainer(model=model,
                        args=training_args,
                        train_dataset=preprocessed_train.shuffle(),
                        eval_dataset=riddleSense_train_map_pre,
                        tokenizer=tokenizer,
                        compute_metrics=None,
                        data_collator=DataCollatorForMultipleChoice(tokenizer))
trainer.train()
# get predictions
predictions, label_ids, _ = trainer.predict(val_pre)
labels = np.array([i for i in map(mapping_func, val_pre['answerKey'])])
predicted_classes = np.argmax(predictions[1], axis=1)
accuracy = np.mean(labels == predicted_classes)
# Accuracy
print(f'Accuracy: {accuracy}')

# This training method trains 5 different models, each for an incrementing number of epochs, this is done so we will have more weight on the earlier epochs
# and as such lower bias in our data map, while still collecting information on sample learnability across epochs.
for i in range(1, 6):
    training_args = TrainingArguments("riddle_sense_check", save_strategy="no", label_names=['answerKey'],
                                      evaluation_strategy="epoch", num_train_epochs=i)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, config=config)
    trainer = CustomTrainer(model=model,
                            args=training_args,
                            train_dataset=preprocessed_train.shuffle(),
                            eval_dataset=riddleSense_train_map_pre,
                            tokenizer=tokenizer,
                            compute_metrics=None,
                            data_collator=DataCollatorForMultipleChoice(tokenizer))
    trainer.train()

# saving correctness and true_label_prob

# uncomment below if you want to save stats, commented to avoid writting over stats


# data_to_dump = {
#     'correctness': torch.mean(torch.stack(correctness), keepdim=True, dim=0),
#     'true_label_probs': torch.stack(true_label_probs,dim=1)

# }
# with open('/stats.pkl', 'wb') as f:
#     pickle.dump(data_to_dump, f)

# load correctness and true_label_prob
with open('/stats.pkl', 'rb') as f:
    data_loaded = pickle.load(f)

# Access
correctness = data_loaded['correctness']
true_label_probs = data_loaded['true_label_probs']
confidence = torch.mean(true_label_probs, dim=1)  # mean prob of TRUE labels across all instances in eval_data
variability = torch.std(true_label_probs, dim=1,
                        correction=0)  # std prob of TRUE labels across all instances in in eval_data

"""#DataMAP visualization"""

# Plot the Data Map.

# Compute bins of correctness
bins = np.digitize(correctness, bins=[0.0000001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Create a dictionary to map bin numbers to bin labels
bin_labels = {0: '0', 1: '(0.0-0.1)', 2: '[0.1-0.2)', 3: '[0.2-0.3)', 4: '[0.3-0.4)', 5: '[0.4-0.5)', 6: '[0.5-0.6)',
              7: '[0.6-0.7)', 8: '[0.7-0.8)', 9: '[0.8-0.9)', 10: '[0.9-1)', 11: '1'}

# Apply the mapping to bins to get bin labels
bins_str = np.vectorize(bin_labels.get)(bins)

colors = px.colors.sample_colorscale("Jet", [n / (11) for n in range(12)])

fig = px.scatter(
    x=variability.numpy().squeeze(),
    y=confidence.numpy().squeeze(),
    color=bins_str.squeeze(),
    symbol=bins_str.squeeze(),
    color_discrete_sequence=colors,
    labels={
        "color": "Correctness",
        "symbol": "Correctness",
        "y": "Confidence",
        "x": "variability"
    },
    category_orders={"color": list(bin_labels.values())[::-1]}
)
fig.show()

import matplotlib.pyplot as plt

confidence_np = confidence.numpy().squeeze()
variability_np = variability.numpy().squeeze()
correctness_np = correctness.numpy().squeeze()

# Create subplots for each histogram
fig, axs = plt.subplots(3, figsize=(10, 15))

# Plot histogram for confidence
axs[0].hist(confidence_np, bins=10, color='blue', alpha=0.7)
axs[0].set_title('Confidence')

# Plot histogram for variability
axs[1].hist(variability_np, bins=10, color='green', alpha=0.7)
axs[1].set_title('Variability')

# Plot histogram for correctness
axs[2].hist(correctness_np, bins=10, color='red', alpha=0.7)
axs[2].set_title('Correctness')

# Display the plots
plt.tight_layout()
plt.show()

"""The scatter plot will contain one point for each evaluation sample.

The x-coordinate of each point will be the variability of the model's predictions for that sample across all epochs, the y-coordinate will be the confidence (the mean model probability of the true label across epochs.) for that sample, and the color and symbol of the point will indicate the bin of the correctness for that sample (how often the model correctly predicted the label of that sample across all epochs).

This visualization will help you to identify patterns in how your model's predictions vary across epochs for different samples. For example, you might see that samples that the model finds harder to classify (those with greater variability and lower confidence) are more likely to be misclassified more often (higher correctness bin).
"""


def classify_points(confidence_threshold=0.2, variability_threshold=0.2):
    hard_to_learn = []
    easy_to_learn = []
    ambiguous = []
    for i, datapoint in enumerate(riddleSense_train_map_pre):
        conf = confidence[i]
        var = variability[i]
        # Hard to learn
        if conf < confidence_threshold and var < variability_threshold:
            hard_to_learn.append(i)

        # Easy to learn
        elif conf >= confidence_threshold and var < variability_threshold:
            easy_to_learn.append(i)
        # Ambigious
        else:
            ambiguous.append(i)

    data_dict = {"easy_to_learn": easy_to_learn,
                 "hard_to_learn": hard_to_learn,
                 "ambiguous": ambiguous}
    return data_dict


data_dict = classify_points(confidence_threshold=0.2, variability_threshold=0.23)

easy_to_learn_idx = data_dict["easy_to_learn"]
hard_to_learn_idx = data_dict["hard_to_learn"]
ambiguous_idx = data_dict["ambiguous"]

# assert result == len(trainset)
print(len(easy_to_learn_idx) + len(hard_to_learn_idx) + len(ambiguous_idx))

easy_to_learn_data = riddleSense_train_map_pre.select(easy_to_learn_idx)
hard_to_learn_data = riddleSense_train_map_pre.select(hard_to_learn_idx)
ambiguous_data = riddleSense_train_map_pre.select(ambiguous_idx)

# print quantity and percentages
total_data_num = riddleSense_train_map_pre.num_rows
print(f"Total datapoints: {total_data_num}")
print(f"Easy datapoints: {easy_to_learn_data.num_rows}")
print(f"Hard datapoints: {hard_to_learn_data.num_rows}")
print(f"Ambiguous datapoints: {ambiguous_data.num_rows}")

easy_data_perc = easy_to_learn_data.num_rows / total_data_num
hard_data_perc = hard_to_learn_data.num_rows / total_data_num
ambiguous_data_perc = ambiguous_data.num_rows / total_data_num

print(f"Easy datapoints percentage: {easy_data_perc}")
print(f"Hard_datapoints percentage: {hard_data_perc}")
print(f"Ambiguous datapoints percentage: {ambiguous_data_perc}")

# Plot the Data Map based on Hard, Easy, Ambiguous regions
categories = np.empty_like(confidence, dtype=np.str)
categories[easy_to_learn_idx] = "easy_to_learn"
categories[hard_to_learn_idx] = "hard_to_learn"
categories[ambiguous_idx] = "ambiguous"

# Now we plot the Data Map
fig = px.scatter(
    x=variability.numpy().squeeze(),
    y=confidence.numpy().squeeze(),
    color=categories,
    labels={
        "color": "Learning Difficulty",
        "y": "Confidence",
        "x": "Variability"
    },
    category_orders={"color": ["Easy_to_learn", "ambiguous", "hard_to_learn"]}
)
fig.show()

config = AutoConfig.from_pretrained("bert-base-uncased")

"""Finetune Bert model only on part of the data (easy/hard/ambigious)"""

true_label_probs = []
correctness = []
acc = []
training_args = TrainingArguments("riddle_sense_check", save_strategy="no", label_names=['answerKey'],
                                  evaluation_strategy="epoch", num_train_epochs=10)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, config=config)
trainer = CustomTrainer(model=model,
                        args=training_args,
                        train_dataset=ambiguous_data,
                        eval_dataset=val_pre,
                        tokenizer=tokenizer,
                        compute_metrics=None,
                        data_collator=DataCollatorForMultipleChoice(tokenizer))

trainer.train()

# plotting
# Create a list of epoch numbers
epoch_numbers = list(range(1, 11))

# Create a DataFrame
df = pd.DataFrame({
    'Epoch': epoch_numbers,
    'Accuracy': acc
})

# Create the plot
fig = px.line(df, x='Epoch', y='Accuracy', title='Accuracy over Epochs')
fig.show()

true_label_probs = []
correctness = []
acc = []
data = easy_to_learn_data
training_args = TrainingArguments("riddle_sense_check", save_strategy="no", label_names=['answerKey'],
                                  evaluation_strategy="epoch", num_train_epochs=10)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, config=config)
trainer = CustomTrainer(model=model,
                        args=training_args,
                        train_dataset=data,
                        eval_dataset=val_pre,
                        tokenizer=tokenizer,
                        compute_metrics=None,
                        data_collator=DataCollatorForMultipleChoice(tokenizer))

trainer.train()

# plotting
# Create a list of epoch numbers
epoch_numbers = list(range(1, 11))

# Create a DataFrame
df = pd.DataFrame({
    'Epoch': epoch_numbers,
    'Accuracy': acc
})

# Create the plot
fig = px.line(df, x='Epoch', y='Accuracy', title='Accuracy over Epochs')
fig.show()

true_label_probs = []
correctness = []
acc = []
training_args = TrainingArguments("riddle_sense_check", save_strategy="no", label_names=['answerKey'],
                                  evaluation_strategy="epoch", num_train_epochs=10)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, config=config)
trainer = CustomTrainer(model=model,
                        args=training_args,
                        train_dataset=concatenate_datasets(
                            [ambiguous_data, ambiguous_data, preprocessed_train]).shuffle(),
                        eval_dataset=val_pre,
                        tokenizer=tokenizer,
                        compute_metrics=None,
                        data_collator=DataCollatorForMultipleChoice(tokenizer))

trainer.train()

# plotting
# Create a list of epoch numbers
epoch_numbers = list(range(1, 11))

# Create a DataFrame
df = pd.DataFrame({
    'Epoch': epoch_numbers,
    'Accuracy': acc
})

# Create the plot
fig = px.line(df, x='Epoch', y='Accuracy', title='Accuracy over Epochs')
fig.show()
