from transformers import AutoTokenizer, BertForMaskedLM
from config import *
from filter_words import filter_words
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
from model import BERTLinear
from torch import optim
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

class Trainer:

    def __init__(self):
        self.domain = config['domain']
        self.bert_type = bert_mapper[self.domain]
        self.device = config['device']
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]

        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        self.model = BERTLinear(self.bert_type, len(
            categories), len(polarities)).to(self.device)

        aspect_dict = {}
        inv_aspect_dict = {}
        for i, cat in enumerate(categories):
            aspect_dict[i] = cat
            inv_aspect_dict[cat] = i

        polarity_dict = {}
        inv_polarity_dict = {}
        for i, pol in enumerate(polarities):
            polarity_dict[i] = pol
            inv_polarity_dict[pol] = i

        self.aspect_dict = aspect_dict
        self.inv_aspect_dict = inv_aspect_dict
        self.polarity_dict = polarity_dict
        self.inv_polarity_dict = inv_polarity_dict

    def load_training_data(self):
        sentences = []
        cats = []
        pols = []
        with open(f'{self.root_path}/label.txt', 'r') as f:
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    cat, pol = line.strip().split()
                    cats.append(self.inv_aspect_dict[cat])
                    pols.append(self.inv_polarity_dict[pol])
                else:
                    sentences.append(line.strip())
        encoded_dict = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt',
            max_length=128,
            return_attention_mask=True,
            truncation=True)
        labels_cat = torch.tensor(cats)
        labels_pol = torch.tensor(pols)
        dataset = TensorDataset(
            labels_cat, labels_pol, encoded_dict['input_ids'], encoded_dict['token_type_ids'], encoded_dict['attention_mask'])
        return dataset

    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def train_model(self, dataset, epochs=epochs):
        self.set_seed(0)
        
        # Prepare dataset
        train_data, val_data = torch.utils.data.random_split(
            dataset, [len(dataset) - validation_data_size, validation_data_size])
        dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        model = self.model
        device = self.device

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in trange(epochs):
            model.train()
            print_loss = 0
            batch_loss = 0
            cnt = 0
            for labels_cat, labels_pol, input_ids, token_type_ids, attention_mask in dataloader:
                optimizer.zero_grad()
                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'token_type_ids': token_type_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }
                loss, _, _ = model(labels_cat.to(device),
                                   labels_pol.to(device), **encoded_dict)
                loss.backward()
                optimizer.step()
                print_loss += loss.item()
                batch_loss += loss.item()
                cnt += 1
                if cnt % 50 == 0:
                    print('Batch loss:', batch_loss / 50)
                    batch_loss = 0

            print_loss /= cnt
            model.eval()
            with torch.no_grad():
                val_loss = 0
                iters = 0
                for labels_cat, labels_pol, input_ids, token_type_ids, attention_mask in val_dataloader:
                    encoded_dict = {
                        'input_ids': input_ids.to(device),
                        'token_type_ids': token_type_ids.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
                    loss, _, _ = model(labels_cat.to(
                        device), labels_pol.to(device), **encoded_dict)
                    val_loss += loss.item()
                    iters += 1
                val_loss /= iters
            # Display the epoch training loss and validation loss
            print("epoch : {:4}/{}, train_loss = {:.6f}, val_loss = {:.6f}".format(
                epoch + 1, epochs, print_loss, val_loss))

    def save_model(self, name):
        torch.save(self.model, f'{self.root_path}/{name}.pth')

    def load_model(self, name):
        self.model = torch.load(f'{self.root_path}/{name}.pth')

    def evaluate(self):
        test_sentences = []
        test_cats = []
        test_pols = []

        with open(f'{self.root_path}/test.txt', 'r') as f:
            for line in f:
                _, cat, pol, sentence = line.strip().split('\t')
                cat = int(cat)
                pol = int(pol)
                test_cats.append(cat)
                test_pols.append(pol)
                test_sentences.append(sentence)

        df = pd.DataFrame(columns=(
            ['sentence', 'actual category', 'predicted category', 'actual polarity', 'predicted polarity']))

        model = self.model
        model.eval()
        device = self.device

        actual_aspect = []
        predicted_aspect = []

        actual_polarity = []
        predicted_polarity = []

        iters = 0
        with torch.no_grad():
            for input, cat, pol in tqdm(zip(test_sentences, test_cats, test_pols)):

                encoded_dict = self.tokenizer([input],
                                              padding=True,
                                              return_tensors='pt',
                                              return_attention_mask=True,
                                              truncation=True).to(device)

                loss, logits_cat, logits_pol = model(torch.tensor([cat]).to(
                    device), torch.tensor([pol]).to(device), **encoded_dict)

                actual_aspect.append(self.aspect_dict[cat])
                actual_polarity.append(self.polarity_dict[pol])

                predicted_aspect.append(
                    self.aspect_dict[torch.argmax(logits_cat).item()])
                predicted_polarity.append(
                    self.polarity_dict[torch.argmax(logits_pol).item()])
                df.loc[iters] = [input, actual_aspect[-1], predicted_aspect[-1],
                                 actual_polarity[-1], predicted_polarity[-1]]
                iters += 1

        df.to_csv(f'{self.root_path}/predictions.csv')

        predicted = np.array(predicted_polarity)
        actual = np.array(actual_polarity)
        print("Polarity")
        print(classification_report(actual, predicted, digits=4))
        print()

        predicted = np.array(predicted_aspect)
        actual = np.array(actual_aspect)
        print("Aspect")
        print(classification_report(actual, predicted, digits=4))
        print()
