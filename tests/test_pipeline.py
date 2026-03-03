import pytest
import torch
from dataset import clean_text, build_vocab, encode, IMDBDataset
from model import BiLSTMClassifier


class TestCleanText:
    def test_strips_html_tags(self):
        assert "<br" not in clean_text("great film<br />loved it")

    def test_lowercases(self):
        assert clean_text("GREAT") == "great"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestBuildVocab:
    def test_pad_and_unk_always_present(self):
        vocab = build_vocab(["hello world"])
        assert vocab["<PAD>"] == 0
        assert vocab["<UNK>"] == 1


class TestEncode:
    def test_output_length_equals_max_len(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
        assert len(encode("hello world", vocab, max_len=10)) == 10


class TestIMDBDataset:
    def test_length(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "good": 2, "bad": 3}
        ds = IMDBDataset(["good film", "bad film"], [1, 0], vocab, max_len=8)
        assert len(ds) == 2


class TestBiLSTMClassifier:
    def test_output_shape(self):
        model = BiLSTMClassifier(vocab_size=100, embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
        x = torch.randint(0, 100, (4, 20))
        out = model(x)
        assert out.shape == torch.Size([4])