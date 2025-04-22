from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

class BertTokenizer():
    def __init__(self, path=None):
        super().__init__()

        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace() # Splitting by whitespace and punctuations

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )

        tokenizer.decoder = decoders.WordPiece()

        if path is not None:
            tokenizer = tokenizer.from_file(path)

        self.tokenizer = tokenizer
    
    def encode(self, sequence):
        return self.tokenizer.encode(sequence)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def train_from_iterator(self, iterator):
        trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

        self.tokenizer.train_from_iterator(iterator, trainer)
