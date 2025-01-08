import io
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import statistics
import math
from util.bleu import get_bleu
BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
def build_vocab(filepath, tokenizer):
        my_counter = Counter()
        with io.open(filepath, encoding="utf8") as filehandle:
            for str in filehandle:
                my_counter.update(tokenizer(str))
        return vocab(my_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
de_vocab = build_vocab('data/train.de', de_tokenizer)
de_vocab.set_default_index(de_vocab['<unk>'])
en_vocab = build_vocab('data/train.en', en_tokenizer)
en_vocab.set_default_index(en_vocab['<unk>'])
#device = 'cpu'

class TranslationDataSet(Dataset):#could be either train, or validation or testing
    def __init__(self, de_file_path, en_file_path) -> None:
        self.PAD_IDX = de_vocab['<pad>']
        self.de_sentences = open(de_file_path, 'r', encoding="utf-8").readlines()
        self.en_sentences = open(en_file_path, 'r', encoding="utf-8").readlines()

    def DEsentenceTokenizer(self, de_sent):
        de_tok_arr = [de_vocab['<bos>']]
        for token in de_tokenizer(de_sent.rstrip("\n")):
            de_tok_arr.append(de_vocab[token])
        de_tok_arr.append(de_vocab['<eos>'])
        return de_tok_arr
    def ENsentenceTokenizer(self, en_sent):
        en_tok_arr = [en_vocab['<bos>']]
        for token in en_tokenizer(en_sent.rstrip("\n")):
            en_tok_arr.append(en_vocab[token])
        en_tok_arr.append(en_vocab['<eos>'])
        return en_tok_arr

    def create_batch(self, each_data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in each_data_batch:
            de_batch.append(de_item)
            en_batch.append(en_item)
        de_batch = pad_sequence(de_batch, padding_value=self.PAD_IDX).to(device)
        en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX).to(device)
        return de_batch, en_batch
        
    def __len__(self):
        return len(self.de_sentences)
        
    def __getitem__(self, idx):
        de_sentence = self.de_sentences[idx]
        en_sentence = self.en_sentences[idx]
        return torch.LongTensor(self.DEsentenceTokenizer(de_sentence)).to(device), torch.LongTensor(self.ENsentenceTokenizer(en_sentence)).to(device)

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class MyTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(MyTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)


    @property
    def device(self):
        return next(self.parameters()).device
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt, PAD_IDX):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
        # Since the memory key padding mask should match the source padding mask in decoder
        memory_key_padding_mask = src_padding_mask.clone()

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask

    def forward(self,
                src: Tensor,
                trg: Tensor,
                PAD_IDX: Tensor):# memory_key_padding_mask deleted
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask = self.create_mask(src, trg, PAD_IDX)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
def plot_loss(avg_batch_loss, avg_batch_loss_val):
    epochs = range(1, len(avg_batch_loss) + 1)
    
    # Plotting batch loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, avg_batch_loss, 'b', label='Training loss')
    plt.plot(epochs, avg_batch_loss_val, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
def plot_bleu(bleu_list):
    epochs = range(1, len(bleu_list) + 1)
    
    # Plotting batch loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, bleu_list, 'b', label='BLEU Score')
    plt.title('BLEU Score for Testing')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU')
    plt.legend()
    plt.show()

def validation_loss(val_iter, model):
    model.eval()
    #validation
    avg_batch_loss_val = 0
    batch_num_val = 0
    for i, data in enumerate(val_iter, 0):
        src = data[0]
        tgt = data[1]
        tgt_input = tgt[:-1, :]#get rid of the last element
        tgt_out = tgt[1:,:]#get rid of the first element

        logits = model(src, tgt_input, torch.tensor(train_dataset.PAD_IDX))
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.PAD_IDX)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        avg_batch_loss_val+=loss.item()
        batch_num_val+=1
    avg_batch_loss_val/=batch_num_val
    return avg_batch_loss_val

max_accumulation_rounds = 1
def train(train_dataset, val_dataset, test_dataset):
    train_iter = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=train_dataset.create_batch)
    val_iter = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=val_dataset.create_batch)
    test_loader = DataLoader(test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            collate_fn=test_dataset.create_batch)
    target_sentences = []
    test_loader_copy = iter(test_loader)
    for _, target in test_loader_copy:
        target = target.to(device)
        for idx, trg_tokens in enumerate(target.T):
            trg_sentence = en_vocab.lookup_tokens(trg_tokens.tolist())
            remove_tokens = {'<unk>', '<pad>', '<bos>', '<eos>'}
            trg_sentence = [word for word in trg_sentence if word not in remove_tokens]
            target_sentences.append(trg_sentence)
    
    SRC_VOCAB_SIZE = len(de_vocab)
    TGT_VOCAB_SIZE = len(en_vocab)
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    EMB_SIZE = 512  # Embedding dimension
    FFN_HID_DIM = 512  # Feedforward dimension
    NHEAD = 8  # Number of attention heads

    model = MyTransformer(NUM_ENCODER_LAYERS,
                        NUM_DECODER_LAYERS,
                        EMB_SIZE, NHEAD,
                        SRC_VOCAB_SIZE,
                        TGT_VOCAB_SIZE,
                        FFN_HID_DIM)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer.zero_grad()

    train_losses = []
    val_losses = []

    old_train_loss = 99999
    new_train_loss = 999
    old_val_loss = 99999
    new_val_loss = 999
    #while True:
    shouldExit = False
    overfitCount = 0
    blue_score_list = []
    while shouldExit == False:
        #train
        avg_batch_loss = 0
        batch_num = 0
        model.train()
        round = 0
        for i, data in enumerate(train_iter, 0):
            round+=1
            src = data[0]
            tgt = data[1]
            tgt = torch.cat((tgt,torch.ones(1,tgt.shape[1]).int().to(device)),axis=0)
            tgt_input = tgt[:-1, :]#get rid of the last element
            tgt_out = tgt[1:,:]#get rid of the first element

            logits = model(src, tgt_input, torch.tensor(train_dataset.PAD_IDX))
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.PAD_IDX)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            #gradient accumulation and update:
            if round>=max_accumulation_rounds:
                optimizer.step()
                avg_batch_loss+=loss.item()
                batch_num+=1
                round = 0
                optimizer.zero_grad()
                # #track 10 round avg validation loss
                # v_loss = validation_loss(val_iter, model)
                # ten_round_val_losses.pop(0)
                # ten_round_val_losses.append(v_loss)
                # old_avg_ten_round_val_loss = new_avg_ten_round_val_loss
                # new_avg_ten_round_val_loss = statistics.mean(ten_round_val_losses)
                # print("train loss:",loss.item(),"avg val loss:", new_avg_ten_round_val_loss, "std:", statistics.stdev(ten_round_val_losses))
                # if old_avg_ten_round_val_loss<new_avg_ten_round_val_loss:
                #     shouldExit = True
                #     break

        #validation
        avg_batch_loss_val = 0
        batch_num_val = 0
        model.eval()
        for i, data in enumerate(val_iter, 0):
            src = data[0]
            tgt = data[1]
            tgt_input = tgt[:-1, :]#get rid of the last element
            tgt_out = tgt[1:,:]#get rid of the first element

            logits = model(src, tgt_input, torch.tensor(train_dataset.PAD_IDX))
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.PAD_IDX)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            avg_batch_loss_val+=loss.item()
            batch_num_val+=1

        avg_batch_loss/=batch_num
        train_losses.append(avg_batch_loss)
        avg_batch_loss_val/=batch_num_val
        val_losses.append(avg_batch_loss_val)
        #determine wheter stop
        old_train_loss = new_train_loss
        new_train_loss = avg_batch_loss
        old_val_loss = new_val_loss
        new_val_loss = avg_batch_loss_val
        if (old_train_loss>new_train_loss and old_val_loss<new_val_loss):
            overfitCount+=1
        else:
            overfitCount = 0
        if overfitCount==2:
            shouldExit = True
        target_sentences_iter = iter(target_sentences)
        blue_score = test_helper(model, test_dataset, device, target_sentences_iter)
        blue_score_list.append(blue_score)
        print("Avg batch loss for Train:", avg_batch_loss, "Avg batch loss for Val:", avg_batch_loss_val, "BLEU score for Test:", blue_score)
    plot_loss(train_losses, val_losses)
    plot_bleu(blue_score_list)
    model = model.to('cpu')
    torch.save(model.state_dict(), 'saved_model.pt')
def test_helper(model, test_dataset, device, target_sentences_iter, max_length=50):
    test_loader = DataLoader(test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            collate_fn=test_dataset.create_batch)
    
    model.eval()
    total_bleu_score = 0
    test_samples = 0

    for source, _ in test_loader:
        source = source.to(device)
        
        # Start with the initial token (assumed to be <bos>)
        tgt_input = torch.full((1, source.shape[1]), en_vocab['<bos>'], device=device)
        complete_sentences = [[] for _ in range(source.shape[1])]

        for _ in range(max_length):
            with torch.no_grad():
                logits = model(source, tgt_input, test_dataset.PAD_IDX)
                # Taking the last predicted token
                last_token_logits = logits[-1, :, :]
                predicted_token = last_token_logits.argmax(dim=-1)
                tgt_input = torch.cat((tgt_input, predicted_token.unsqueeze(0)), dim=0)

                # Check if all sequences predict the <eos> token
                if (predicted_token == en_vocab['<eos>']).all():
                    break

            for i, token_id in enumerate(predicted_token):
                if token_id != en_vocab['<eos>']:
                    complete_sentences[i].append(token_id.item())

        for idx, (pred_tokens, trg_sentence) in enumerate(zip(complete_sentences, target_sentences_iter)):
            pred_sentence = en_vocab.lookup_tokens(pred_tokens)
            bleu_score = get_bleu(hypotheses=pred_sentence, reference=trg_sentence)
            total_bleu_score += bleu_score
            test_samples += 1

    average_bleu_score = total_bleu_score / test_samples
    return average_bleu_score
def test(model_dir, test_dataset:TranslationDataSet, device, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, max_length=50):
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    EMB_SIZE = 512  # Embedding dimension
    FFN_HID_DIM = 512  # Feedforward dimension
    NHEAD = 8  # Number of attention heads
    model = MyTransformer(NUM_ENCODER_LAYERS,
                        NUM_DECODER_LAYERS,
                        EMB_SIZE, NHEAD,
                        SRC_VOCAB_SIZE,
                        TGT_VOCAB_SIZE,
                        FFN_HID_DIM)
    model.load_state_dict(torch.load(model_dir))

    test_loader = DataLoader(test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            collate_fn=test_dataset.create_batch)
    
    model.eval()
    total_bleu_score = 0
    test_samples = 0

    target_sentences = []
    test_loader_copy = iter(test_loader)
    for _, target in test_loader_copy:
        target = target.to(device)
        for idx, trg_tokens in enumerate(target.T):
            trg_sentence = en_vocab.lookup_tokens(trg_tokens.tolist())
            remove_tokens = {'<unk>', '<pad>', '<bos>', '<eos>'}
            trg_sentence = [word for word in trg_sentence if word not in remove_tokens]
            target_sentences.append(trg_sentence)
        
    target_sentences_iter = iter(target_sentences)
    for source, _ in test_loader:
        source = source.to(device)
        
        # Start with the initial token (assumed to be <bos>)
        tgt_input = torch.full((1, source.shape[1]), en_vocab['<bos>'], device=device)
        complete_sentences = [[] for _ in range(source.shape[1])]

        for _ in range(max_length):
            with torch.no_grad():
                logits = model(source, tgt_input, test_dataset.PAD_IDX)
                # Taking the last predicted token
                last_token_logits = logits[-1, :, :]
                predicted_token = last_token_logits.argmax(dim=-1)
                tgt_input = torch.cat((tgt_input, predicted_token.unsqueeze(0)), dim=0)

                # Check if all sequences predict the <eos> token
                if (predicted_token == en_vocab['<eos>']).all():
                    break

            for i, token_id in enumerate(predicted_token):
                if token_id != en_vocab['<eos>']:
                    complete_sentences[i].append(token_id.item())

        for idx, (pred_tokens, trg_sentence) in enumerate(zip(complete_sentences, target_sentences_iter)):
            pred_sentence = en_vocab.lookup_tokens(pred_tokens)
            bleu_score = get_bleu(hypotheses=pred_sentence, reference=trg_sentence)
            total_bleu_score += bleu_score
            test_samples += 1

    average_bleu_score = total_bleu_score / test_samples
    print(f"Average BLEU Score: {average_bleu_score}")

if __name__ == "__main__":
    # Create dataset
    train_dataset = TranslationDataSet('data/train.de', 'data/train.en')
    val_dataset = TranslationDataSet('data/val.de', 'data/val.en')
    test_dataset = TranslationDataSet('data/test.de', 'data/test.en')

    train(train_dataset, val_dataset, test_dataset)
    SRC_VOCAB_SIZE = len(de_vocab)
    TGT_VOCAB_SIZE = len(en_vocab)
    test('saved_model.pt', test_dataset,'cpu', SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)