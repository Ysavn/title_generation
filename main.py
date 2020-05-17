import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, BucketIterator
from torchtext import data
from sklearn.utils import shuffle
from gensim.summarization import summarize

import re, json
import spacy
import numpy as np
import pandas as pd
import argparse, copy
import random
import math
import time, datetime

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim  # dim of hidden & cell states
        self.n_layers = n_layers  # num of layers in LSTM

        # emb_dim is dimension of embedding vectors
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # dropout is applied to hidden states between the layers of LSTM
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim  # size of target vocabulary among which to predict
        self.hid_dim = hid_dim  # dimension for hidden & cell states
        self.n_layers = n_layers  # number of layers for decoder (same as Encoder)

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalizeString(sent):
    '''
    Minimal natural language cleansing and preprocessing
    '''
    sent = sent.strip()
    sent = re.sub(r"([.,?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z.,?-]+", r" ", sent)

    # differentiate '- M' from '- m' with '$- M'
    sent = re.sub(r"(- [A-Z])", r"$\1", sent)
    # smooth '$- M' to ' M' eg Exponential- Multiplication
    sent = sent.replace("$-", "")
    # smooth '- m' to 'm' eg  multi- plicative
    sent = sent.replace("- ", "")
    # now leftover genuine cases are handled eg non-negative
    sent = sent.replace("-", " ")
    return sent.lower()


def shorten_abstract(abstract):
    '''
    performs extractive summarization on abstract for faster training,
    has hardcoded limits within this module
    '''
    if len(abstract.split()) > 300:
        abstract = summarize(abstract)

    return abstract


def load_data(path):
    '''
	returns data as a dataframe of tilte and abstract, irrespective of source
	'''
    data = list()

    if '.json' in path:
        with open(path) as f:
            data = json.load(f)
            data = pd.DataFrame(data)
            data = data[['summary', 'title']]
            data.columns = ['abstract', 'title']
    elif '.csv' in path:
        data = pd.read_csv(path)
        data = data[data['abstract'] != 'Abstract Missing'][['title', 'abstract']]
    else:
        raise Exception('Invalid data source')

    return data


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def construct_dataset(in_data):
    examples = []
    fields = [("src", SRC), ("trg", TRG)]
    for idx, row in in_data.iterrows():
        title = normalizeString(row['title'])
        abstract = shorten_abstract(normalizeString(row['abstract']))
        examples.append(data.Example.fromlist([abstract, title], fields))
    return data.Dataset(examples, fields)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def calcRogue(actual_titles, pred_titles):
    rogue = 0
    for i in range(len(actual_titles)):
        mp = {}
        for num in actual_titles[i]:
            if num in mp:
                mp[num] += 1
            else:
                mp[num] = 1
        cnt = 0
        for num in pred_titles[i]:
            if num in mp:
                mp[num] -= 1
                cnt += 1
                if mp[num] == 0:
                    del mp[num]
        recall = cnt / len(actual_titles[i])
        rogue += recall
    return rogue * 100 / len(actual_titles)


def evaluate(model, iterator, criterion, is_test):
    model.eval()

    epoch_loss = 0
    pred_titles = []
    actual_titles = []

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            tmp_output = output
            tmp_trg = trg
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            if is_test:
                for j in range(tmp_trg.shape[1]):
                    trg_val = []
                    pred_val = []
                    for k in range(tmp_trg[:, j].shape[0]):
                        id = tmp_output[k, j].argmax().item()
                        pred_val.append(id)
                        trg_val.append(tmp_trg[k, j].item())
                    pred_titles.append(pred_val)
                    actual_titles.append(trg_val)

    if is_test:
        rogue = calcRogue(actual_titles, pred_titles)
        return epoch_loss / len(iterator), rogue

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    SRC = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    input_data1 = load_data("/home/csci5980/saluj012/S2S_1/papers.csv")
    input_data2 = load_data("/home/csci5980/saluj012/S2S_1/arxivData.json")
    input_data = shuffle(pd.concat([input_data1, input_data2]))
    N = input_data.shape[0]
    train_data = input_data.head(36000)
    tmp = input_data.tail(N - 36000)
    N_tmp = tmp.shape[0]
    val_data = tmp.head(8800)
    test_data = input_data.tail(N_tmp - 8800)
    # print(train_data.shape, val_data.shape, test_data.shape)

    spacy_en = spacy.load('en')
    train_dataset = construct_dataset(train_data)
    val_dataset = construct_dataset(val_data)
    test_dataset = construct_dataset(test_data)

    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128

    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.src),
        device=device)

    # print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
    # print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 4
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    
    N_EPOCHS = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="set number of epochs")
    parser.add_argument("--weight_decay", help="l2 regularization")
    parser.add_argument("--lr", help="learning rate")
    args = parser.parse_args()
    if args.epochs:
        N_EPOCHS = int(args.epochs)
    if args.lr:
        lr = float(args.lr)
    if args.weight_decay:
        weight_decay = float(args.weight_decay)
    print("Epochs: {}\nLR: {}\nWD: {}".format(N_EPOCHS, lr, weight_decay))

    CLIP = 1
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    best_valid_loss = float('inf')


    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_iterator, criterion, 0)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())            

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    best_valid_loss = float('inf')
    date_time = str(datetime.datetime.now())
    date_time = date_time.split(" ")
    curr_time = date_time[1][0:date_time[1].rfind(":")+3]
    print("Current Time: ", curr_time)

    torch.save(best_model_wts, '/home/csci5980/saluj012/S2S_1/' + 'model_s2s_' + str(curr_time) + '.pt')

    model.load_state_dict(torch.load('/home/csci5980/saluj012/S2S_1/' + 'model_s2s_' + str(curr_time) + '.pt'))

    test_loss, rogue_metric = evaluate(model, test_iterator, criterion, 1)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Rogue Metric: {rogue_metric:.2f}')

