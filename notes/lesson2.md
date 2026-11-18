# Lesson 2 - Text to tokens

- A network needs numbers, so text becomes integer ids. The simplest scheme is character-level: sort the unique characters and give each an id. English text gives about 65 ids and nothing to train.
- Corpus: tiny-shakespeare, 1,115,394 characters, vocabulary of 65. It is downloaded once into the git-ignored `data/` folder.
- `CharTokenizer` provides `encode` and `decode`; decoding an encoded string gives back the original.
- The whole corpus becomes one tensor, with the last 10% held out for validation: 1,003,854 train and 111,540 validation tokens.
- Real GPTs use byte-pair encoding: start from characters and repeatedly merge the most frequent adjacent pair. Sequences get much shorter and nothing is "unknown". Only the vocabulary changes; the rest of the pipeline is the same.
- Code: `lessons/lesson2/data.py` (tokenizer and batching, shared by later lessons) and `lesson2.py`.
