"""Caption cleaning, tokenization, and vocabulary; no dataset needed."""
from src.data.text import Vocabulary, split_description, tokenize


def test_split_description_splits_lowercases_and_strips():
    description = "This is a Bird.\n  A yellow bird.  \n"
    assert split_description(description) == ["this is a bird.", "a yellow bird."]


def test_split_description_drops_empty_lines():
    description = "a brown bird\n\n   \nit has a long tail"
    assert split_description(description) == ["a brown bird", "it has a long tail"]


def test_split_description_removes_duplicates_keeping_order():
    description = "a brown bird\na yellow bird\na brown bird"
    assert split_description(description) == ["a brown bird", "a yellow bird"]


def test_split_description_none_returns_empty_list():
    assert split_description(None) == []
    assert split_description("") == []


def test_split_description_replaces_corrupted_character_with_space():
    assert split_description("this��bird")[0] == "this bird"


def test_split_description_normalises_comma_spacing():
    assert split_description("breast,vent")[0] == "breast, vent"
    assert split_description("belly , breast")[0] == "belly, breast"


def test_tokenize_lowercases_and_splits_words():
    assert tokenize("This Bird has 2 wings.") == ["this", "bird", "has", "2", "wings"]


def test_tokenize_keeps_contractions_together():
    assert tokenize("it's a bird") == ["it's", "a", "bird"]


def test_vocabulary_special_tokens_have_fixed_ids():
    vocab = Vocabulary()
    assert vocab.stoi["<pad>"] == 0
    assert vocab.stoi["<bos>"] == 1
    assert vocab.stoi["<eos>"] == 2
    assert vocab.stoi["<unk>"] == 3


def test_vocabulary_build_respects_min_freq():
    captions = ["red bird", "red wing", "blue wing"]
    vocab = Vocabulary(min_freq=2).build(captions)

    assert "red" in vocab.stoi
    assert "wing" in vocab.stoi
    assert "blue" not in vocab.stoi


def test_vocabulary_encode_adds_bos_and_eos():
    vocab = Vocabulary(min_freq=1).build(["red bird"])
    ids = vocab.encode("red bird")

    assert ids[0] == vocab.stoi["<bos>"]
    assert ids[-1] == vocab.stoi["<eos>"]
    assert ids == [vocab.stoi["<bos>"], vocab.stoi["red"], vocab.stoi["bird"], vocab.stoi["<eos>"]]


def test_vocabulary_encode_maps_unknown_words_to_unk():
    vocab = Vocabulary(min_freq=1).build(["red bird"])
    ids = vocab.encode("a green bird")

    unk = vocab.stoi["<unk>"]
    assert ids == [vocab.stoi["<bos>"], unk, unk, vocab.stoi["bird"], vocab.stoi["<eos>"]]


def test_vocabulary_decode_round_trips_and_stops_at_eos():
    vocab = Vocabulary(min_freq=1).build(["red bird", "blue wing"])
    caption = "red bird"

    assert vocab.decode(vocab.encode(caption)) == caption


def test_vocabulary_decode_stops_before_padding_after_eos():
    vocab = Vocabulary(min_freq=1).build(["red bird"])
    ids = vocab.encode("red bird") + [vocab.stoi["<pad>"], vocab.stoi["<pad>"]]

    assert vocab.decode(ids) == "red bird"
