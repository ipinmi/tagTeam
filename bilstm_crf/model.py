# @Author: Chibundum Adebayo

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchcrf import CRF as TorchCRF


class Char_Word_BiLSTM(nn.Module):
    def __init__(
        self,
        params,
    ):
        super(Char_Word_BiLSTM, self).__init__()
        self.use_char = params.use_char
        self.drop_out = nn.Dropout(p=params.dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.char_hidden_dim = params.char_hidden_dim // 2
        self.word_hidden_dim = params.word_hidden_dim // 2

        # Load the pretrained embeddings if True
        if params.use_pretrained:
            self.pretrained_word_embedding = torch.load(params.pretrained_emd_path)

            self.word_embeddings = nn.Embedding.from_pretrained(
                self.pretrained_word_embedding, freeze=False
            )
        else:
            # intialize the word level embeddings
            self.word_embeddings = nn.Embedding(
                num_embeddings=params.word_vocab_size,
                embedding_dim=params.word_emb_dim,
                padding_idx=params.padding_index,
            )
            # Initialized with Xavier uniform distribution.
            torch.nn.init.xavier_uniform_(self.word_embeddings.weight)

        if self.use_char:
            # Intialize the character level embeddings
            self.char_embeddings = nn.Embedding(
                num_embeddings=params.char_vocab_size,
                embedding_dim=params.char_emb_dim,
                padding_idx=params.padding_index,
            )
            # Initialized with Xavier uniform distribution.
            torch.nn.init.xavier_uniform_(self.char_embeddings.weight)

            # Initialized the char level LSTM
            self.char_LSTM = nn.LSTM(
                input_size=params.char_emb_dim,
                hidden_size=self.char_hidden_dim,
                num_layers=params.num_lstm_layer,
                bidirectional=True,
                batch_first=True,
            )

            # Initialize the word level LSTM WITH character level information
            self.word_LSTM = nn.LSTM(
                input_size=params.word_emb_dim + params.char_hidden_dim,
                hidden_size=self.word_hidden_dim,
                num_layers=params.num_lstm_layer,
                bidirectional=True,
                batch_first=True,
            )
        else:
            # Initialize the word level LSTM WITHOUT character level information
            self.word_LSTM = nn.LSTM(
                input_size=params.word_emb_dim,
                hidden_size=self.word_hidden_dim,
                num_layers=params.num_lstm_layer,
                bidirectional=True,
                batch_first=True,
            )

        # Linear projection to tagset size
        self.hidden2tag = nn.Linear(self.word_hidden_dim * 2, params.tag_vocab_size)
        nn.init.xavier_normal_(self.hidden2tag.weight.data)
        nn.init.normal_(self.hidden2tag.bias.data)

    def forward(
        self,
        word_seq: torch.Tensor,
        word_len: int,
        char_seq: torch.Tensor,
        char_len: list,
        char_recover: dict,
    ):
        # word embedded: [sequence_length, embedding_dimension]
        embedded_word = self.word_embeddings(word_seq)

        if self.use_char:
            # embedded_chars: [sequence_length, word_lengths, char_embedding_dim]
            embedded_char = self.char_embeddings(char_seq).transpose(0, 1)
            embedded_char = self.drop_out(embedded_char)

            # To ensure that the LSTM only sees the non-padded items
            packed_embedded_char = nn.utils.rnn.pack_padded_sequence(
                embedded_char, char_len
            )

            char_lstm_output, _ = self.char_LSTM(packed_embedded_char)

            char_output, char_output_lens = nn.utils.rnn.pad_packed_sequence(
                char_lstm_output
            )
            char_output = char_output.transpose(0, 1)

            temp_char_lvl_features = Variable(
                torch.FloatTensor(
                    torch.zeros((char_output.size(0), char_output.size(2)))
                )
            ).to(self.device)

            # Concatenate the last hidden state of forward and backward LSTM
            for i, index in enumerate(char_output_lens):
                temp_char_lvl_features[i] = torch.cat(
                    (
                        char_output[i, index - 1, : self.char_hidden_dim],
                        char_output[i, 0, self.char_hidden_dim :],
                    )
                )
            char_lvl_features = temp_char_lvl_features.clone()

            # Reorder the char_lvl_features to match the word sequence
            for i in range(char_lvl_features.size(0)):
                char_lvl_features[char_recover[i]] = temp_char_lvl_features[i]

            # [sequence_length, word_embedding_dim + char_last_hidden_dim]
            char_Word_Embeddings = torch.cat((embedded_word, char_lvl_features), 1)
        else:
            # Use only the word embeddingss
            # [sequence_length, word_embedding_dim]
            char_Word_Embeddings = embedded_word.clone()

        char_Word_Embeddings = char_Word_Embeddings.unsqueeze(1)
        char_Word_Embeddings = self.drop_out(char_Word_Embeddings)

        combined_lstm_output, _ = self.word_LSTM(char_Word_Embeddings)

        combined_lstm_output = combined_lstm_output.view(
            word_len, self.word_hidden_dim * 2
        )
        lstm_out = self.drop_out(combined_lstm_output)
        final_output = self.hidden2tag(lstm_out)

        return final_output


class POS_Tagger(nn.Module):
    def __init__(
        self,
        params,
    ):
        super(POS_Tagger, self).__init__()
        self.crf = params.with_crf
        self.tag_vocab_size = params.tag_vocab_size

        self.feature_extractor = Char_Word_BiLSTM(params=params)

        if self.crf:
            self.tagger = TorchCRF(self.tag_vocab_size)
        else:
            self.tagger = torch.nn.Softmax(dim=-1)
            self.loss_function = torch.nn.CrossEntropyLoss(
                ignore_index=params.padding_index
            )

    def loss_function(
        self,
        input_seq: torch.Tensor,
        input_len: int,
        char_input_seq: torch.Tensor,
        char_input_len: list,
        char_recover: dict,
        gold_tag_seq: torch.Tensor,
    ):
        output_features = self.feature_extractor(
            input_seq, input_len, char_input_seq, char_input_len, char_recover
        )
        if self.crf:
            batch_size = 1  # Set to 1 because we are processing one sentence at a time
            # To ensure that the dimensions are (seq_length, batch_size, tag_vocab_size)
            reshaped_output_features = output_features.view(
                input_len, batch_size, self.tag_vocab_size
            )
            reshaped_tag_seq = gold_tag_seq.view(input_len, batch_size)

            # Negated because the library provides log likelihood
            loss = -self.tagger.forward(
                reshaped_output_features,
                reshaped_tag_seq,
                mask=None,
                reduction="mean",
            )
        else:
            gold_tag_seq = Variable(gold_tag_seq)
            loss = nn.functional.cross_entropy(output_features, gold_tag_seq)
        return loss

    def forward(
        self,
        input_seq: torch.Tensor,
        input_len: list,
        char_input_seq: torch.Tensor,
        char_input_len: list,
        char_recover: dict,
        tag_seq: torch.Tensor,
    ):
        output_features = self.feature_extractor(
            input_seq, input_len, char_input_seq, char_input_len, char_recover
        )

        if self.crf:
            batch_size = 1  # Set to 1 because we are processing one sentence at a time
            # Ensure the dimensions are (seq_length, batch_size, tag_vocab_size)
            reshaped_output_features = output_features.view(
                input_len, batch_size, self.tag_vocab_size
            )
            output_tags = self.tagger.decode(reshaped_output_features)
            score = None  # to match the return type of the non-crf block
            return score, output_tags
        else:
            score, output_tags = torch.max(output_features, 1)
            output_tags = output_tags.cpu().tolist()

            return score, output_tags
