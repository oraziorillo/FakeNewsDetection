import transformers
from transformers import BertModel, BertTokenizer, AdamW

import torch
import numpy as np
import pandas as pd
import warnings

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

global PRE_TRAINED_MODEL_NAME

def to_sentiment(rating):
  rating = int(rating)
  if rating <= 2:
    return 0
  elif rating == 3:
    return 1
  else: 
    return 2

class ReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      truncation=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

class SentimentBERT(nn.Module):

  def __init__(self, n_classes):
    super(SentimentBERT, self).__init__()

    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.dropout = nn.Dropout(p=0.5)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    
    bert_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    pooled_output  = bert_output[1]
    dropout_output = self.dropout(pooled_output)
    return self.out(dropout_output)

def train_model(model, data_loader, loss_fn, optimizer, device, n_examples):
  
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    correct_predictions += torch.sum(preds == targets)

    loss = loss_fn(outputs, targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  
  return correct_predictions.double() / n_examples, np.mean(losses)

def predict_model(model, data_loader):
  model = model.eval()
  
  predictions = []
  targets = []

  with torch.no_grad():
    for d in data_loader:

      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      target = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      predictions.extend(preds)
      targets.extend(target)

  predictions = torch.stack(predictions).cpu()
  targets = torch.stack(targets).cpu()
  return predictions, targets


if __name__ == '__main__':
  
  warnings.filterwarnings("ignore", category=FutureWarning)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')
  RANDOM_SEED = 1
  BATCH_SIZE = 16
  MAX_LEN = 128
  EPOCHS = 10
  
  reviews = pd.read_csv("data/reviews.csv")
  reviews['sentiment'] = reviews.score.apply(to_sentiment)

  reviews_train, reviews_test = train_test_split(reviews, test_size=0.3, random_state=RANDOM_SEED)

  reviews_train_dataset = ReviewDataset(
    reviews=reviews_train.content.to_numpy(),
    targets=reviews_train.sentiment.to_numpy(),
    tokenizer=TOKENIZER,
    max_len=MAX_LEN
  )

  reviews_test_dataset = ReviewDataset(
    reviews=reviews_test.content.to_numpy(),
    targets=reviews_test.sentiment.to_numpy(),
    tokenizer=TOKENIZER,
    max_len=MAX_LEN
  )

  train_data_loader = DataLoader(
    reviews_train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2
  )

  test_data_loader = DataLoader(
    reviews_test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2
  )

  model = SentimentBERT(3)
  model = model.to(device)

  data = next(iter(train_data_loader))
  input_ids = data['input_ids'].to(device)
  attention_mask = data['attention_mask'].to(device)

  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  loss_fn = nn.CrossEntropyLoss().to(device)


  for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 30)

    train_acc, train_loss = train_model(
      model,
      train_data_loader,    
      loss_fn, 
      optimizer, 
      device,
      len(reviews_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

  
  y_pred, y_test = predict_model(
    model,
    test_data_loader
  )

  class_names = ['negative', 'neutral', 'positive']
  print(classification_report(y_test, y_pred, target_names=class_names))

  torch.save(model.state_dict(), 'models/state_dict_sentbert.pt')

  # model = SentimentBERT(3)
  # model.load_state_dict(torch.load('models/state_dict_sentbert.pt', map_location=torch.device('cpu')))
  # model.eval()




