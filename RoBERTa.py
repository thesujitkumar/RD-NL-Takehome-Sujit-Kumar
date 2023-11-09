import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class input_data(Dataset):
    def __init__(self, text, labels, tokenizer, max_length):
        """
        Argumnets:
            data (list): input text samples
            labels (list): labels text is generated  by boat or human.
            tokenizer: Pre-trained tokenizer (e.g., from the 'transformers' library).
            max_length (int): Maximum sequence length for tokenization.
        """
        self.data = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(str(text), padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

# Read the input data
train =  pd.read_csv("train_bin.csv")
print(len(train))
valid =  pd.read_csv("dev_bin.csv")
print(len(valid ))
test=  pd.read_csv("test_bin.csv")
print(len(test ))

# convert dataframe columns to list
data = train['text'].values.tolist()

labels = train['class'].values.tolist()

batch_size = 2
# Intialize the tokenizor
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# Create an instance of the custom dataset
train_dataset = input_data(data, labels, tokenizer, max_length=128)

# Create a data loader for the custom dataset
# batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(len(train_loader))

data = valid['text'].values.tolist()
labels = valid['class'].values.tolist()
valid_dataset = input_data(data, labels, tokenizer, max_length=128)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
print(len(valid_loader))

data = test['text'].values.tolist()
labels = test['class'].values.tolist()
test_dataset = input_data(data, labels, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(len(test_loader))

# Initialize pretraine RoBERTa model for doccument classification
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)  # 2 for binary classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

#Functions for model performance evaluations
def my_score(y_test, y_pred):
    acc =accuracy_score(y_test, y_pred)
    f1= f1_score(y_test, y_pred, average='macro')

    f1_classwise = f1_score(y_test, y_pred, average=None)


    return acc,f1,f1_classwise


# Training of model
best=0.0
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(batch)

    # Evalaute the performance of model on validation set
    model.eval()
    val_predictions = []
    val_true_labels = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            val_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")


    if val_accuracy > best:
        model.save_pretrained("best_model")


"Load the best models and evalaute on test set"
best_model = RobertaForSequenceClassification.from_pretrained("best_model")
best_model.to(device)
best_model.eval()


test_predictions = []
test_true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        output = best_model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        test_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_true_labels, test_predictions)

print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, test_accuracy: {test_accuracy * 100:.2f}%")
print("performance {acc} : Fmeasure: classwise_score")
score= my_score(test_true_labels, test_predictions)
print(score)
