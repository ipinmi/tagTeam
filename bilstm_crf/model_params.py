# @Author: Chibundum Adebayo


class Parameters:
    def __init__(self, text_obj, char_text_obj, tag_obj, emd_path):
        self.lang_code = "zh"  # "de"/ "zh" / "af" / "en"
        self.pretrained_emd_path = emd_path

        self.char_vocab = char_text_obj
        self.word_vocab = text_obj
        self.tag_vocab = tag_obj

        self.word_vocab_size = self.word_vocab.idx2token.__len__()
        self.char_vocab_size = self.char_vocab.idx2token.__len__()
        self.word_emb_dim = 100
        self.char_emb_dim = 30
        self.tag_vocab_size = self.tag_vocab.idx2token.__len__()

        self.start_tag = "<sos>"
        self.stop_tag = "<eos>"

        self.padding_index = text_obj.token2idx["<pad>"]

        # Architectural Decisions
        self.use_pretrained = False
        self.use_char = True
        self.with_crf = False

        ## Training Hyperparameters
        self.batch_size = 10
        self.char_hidden_dim = 25  # 30
        self.word_hidden_dim = 100  # 200
        self.dropout = 0.5  # 0.5
        self.num_lstm_layer = 1

        self.num_epochs = 10
        self.gpu = True
        self.learning_rate = 0.001  # 0.015
        self.lr_decay = 0.05
        self.clip = 5.0
        self.momentum = 0.9
