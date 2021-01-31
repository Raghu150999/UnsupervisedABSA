# UnsupervisedABSA
We propose a BERT based three-step mixed semi-supervised model, which jointly detects aspect and sentiment in a given review sentence. The first step takes a small set of seed words for each aspect and each sentiment class to construct class vocabulary for each class using a context-aware BERT masked language model. The second step extracts aspect/opinion term(s) using POS tags and constructed vocabularies in step one. In the last step, extracted aspect and opinion words are used as label data to train a BERT based joint deep neural network for aspect and sentiment classification.

## CASC
In this work, we leverage power of post-trained, domain knowledge BERT (DK-BERT) and present a simple and highly efficient semi-supervised hybrid CASC approach for Context aware Aspect category and Sentiment Classification. Our model is built in a simple three-step process: 
1. We take a small set of seed words for each aspect and each sentiment class and then construct class vocabulary for each class that contains semantically coherent words with the seed words using BERT masked language model (MLM).
2. We take unlabeled training corpus and extract potential aspects and opinion terms using POS tags and class vocabularies constructed in the previous step.
3. In the last step, We make use of extracted aspect and opinion term as label data and jointly train BERT based neural model for aspect and sentiment classification.

<img src="Framework.png" width=1200>

### Requirements

To install all dependencies:
```bash
pip install -r requirements.txt
```

### Quick start
Run command:
```bash
python main.py
```

### Datasets
Prepared datasets for both `laptop` and `restaurant` domain are available under `datasets/` directory, acquired from [Huang et al.](https://github.com/teapot123/JASen).

### Configuration
All configuration and model hyperparameters can be found at `config.py`.
