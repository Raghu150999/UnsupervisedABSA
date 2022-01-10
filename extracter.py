from tqdm import tqdm
from config import *
import spacy


class Extracter:
    '''
    Extract potential-aspects and potential-opinion words
    '''

    def __init__(self):
        self.smodel = spacy.load('en')
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]

    def __call__(self):
        # Extract potential-aspects and potential-opinions
        sentences = []
        aspects = []
        opinions = []

        with open(f'{self.root_path}/train.txt') as f:
            for line in tqdm(f):
                text = line.strip()
                sentences.append(text)
                words = self.smodel(text)
                o = []
                a = []
                for word in words:
                    if word.tag_.startswith('JJ') or word.tag_.startswith('RR'):
                        # Adjective or Adverb
                        o.append(word.text)
                    if word.tag_.startswith('NN'):
                        # Noun
                        a.append(word.text)
                opinions.append(' '.join(o) if len(o) > 0 else '##')
                aspects.append(' '.join(a) if len(a) > 0 else '##')

        return sentences, aspects, opinions
