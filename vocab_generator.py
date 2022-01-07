from transformers import AutoTokenizer, BertForMaskedLM
from config import *
from filter_words import filter_words
import torch
from tqdm import tqdm

class VocabGenerator:

    def __init__(self, save_results=True):
        self.domain = config['domain']
        self.bert_type = bert_mapper[self.domain]
        self.device = config['device']
        self.mlm_model = BertForMaskedLM.from_pretrained(self.bert_type).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]
        self.save_results = save_results
    
    def __call__(self):
        aspect_categories = aspect_category_mapper[self.domain]
        aspect_seeds = aspect_seed_mapper[self.domain]
        aspect_vocabularies = self.generate_vocabularies(aspect_categories, aspect_seeds)

        sentiment_categories = sentiment_category_mapper[self.domain]
        sentiment_seeds = sentiment_seed_mapper[self.domain]
        sentiment_vocabularies = self.generate_vocabularies(sentiment_categories, sentiment_seeds)

        return aspect_vocabularies, sentiment_vocabularies

    def generate_vocabularies(self, categories, seeds):
        # Initialise empty frequency table
        freq_table = {}
        for cat in categories:
            freq_table[cat] = {}
        
        # Populate vocabulary frequencies for each category
        for category in categories:
            print(f'Generating vocabulary for {category} category...')
            with open(f'{self.root_path}/train.txt') as f:
                for line in tqdm(f):
                    text = line.strip()
                    if category in text:
                        ids = self.tokenizer(text, return_tensors='pt', truncation=True)['input_ids']
                        tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
                        word_predictions = self.mlm_model(ids.to(self.device))[0]
                        word_scores, word_ids = torch.topk(word_predictions, K_1, -1)
                        word_ids = word_ids.squeeze(0)
                        for idx, token in enumerate(tokens):
                            if token in seeds[category]:
                                self.update_table(freq_table, category, self.tokenizer.convert_ids_to_tokens(word_ids[idx]))
        
        # Remove words appearing in multiple vocabularies (generate disjoint sets)
        for category in categories:
            for key in freq_table[category]:
                for cat in categories:
                    if freq_table[cat].get(key) != None and freq_table[cat][key] < freq_table[category][key]:
                        del freq_table[cat][key]
        
        vocabularies = {}

        for category in categories:
            words = []
            for key in freq_table[category]:
                words.append((freq_table[category][key], key))
            words.sort(reverse=True)
            vocabularies[category] = words

            if self.save_results:
                # Saving vocabularies
                f = open(f'{self.root_path}/dict_{category}.txt', 'w')
                for freq, word in words:
                    f.write(f'{word} {freq}\n')
                f.close()

        return vocabularies
    
    def update_table(self, freq_table, cat, tokens):
        for token in tokens:
            if token in filter_words or '##' in token:
                continue
            freq_table[cat][token] = freq_table[cat].get(token, 0) + 1



    
