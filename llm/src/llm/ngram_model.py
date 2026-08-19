import random
from collections import defaultdict
from collections.abc import Mapping, Sequence


class NGramModel:
    """A word-level model containing unigram, bigram, and trigram counts."""

    def __init__(self) -> None:
        self.unigrams: dict[str, int] = {}
        self.bigrams: dict[str, dict[str, int]] = {}
        self.trigrams: dict[tuple[str, str], dict[str, int]] = {}

    def train(self, words: Sequence[str]) -> None:
        """Train the model on an ordered sequence of word tokens.

        Calling this method replaces counts from any previous training run.

        Args:
            words: Tokens in the order in which they appear in the corpus.
        """
        unigram_counts: defaultdict[str, int] = defaultdict(int)
        bigram_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        trigram_counts: defaultdict[
            tuple[str, str], defaultdict[str, int]
        ] = defaultdict(lambda: defaultdict(int))

        for index, word in enumerate(words):
            unigram_counts[word] += 1

            if index >= 1:
                previous_word = words[index - 1]
                bigram_counts[previous_word][word] += 1

            if index >= 2:
                context = (words[index - 2], words[index - 1])
                trigram_counts[context][word] += 1

        self.unigrams = dict(unigram_counts)
        self.bigrams = {
            context: dict(counts)
            for context, counts in bigram_counts.items()
        }
        self.trigrams = {
            context: dict(counts)
            for context, counts in trigram_counts.items()
        }

    def predict(
        self,
        first_word: str | None = None,
        second_word: str | None = None,
        temperature: float = 0.8,
    ) -> str:
        """Predict a next word by sampling from the best available n-gram.

        A known two-word context uses trigram counts. If that context is not
        known, the most recent supplied word is tried as a bigram context.
        The method falls back to unigram frequencies when neither is known.

        Args:
            first_word: The older, optional context word.
            second_word: The newer, optional context word.
            temperature: Sampling temperature. Lower values favor common words;
                higher values increase variety. Must be greater than zero.

        Returns:
            A sampled prediction for the next word.

        Raises:
            RuntimeError: If the model has not been trained on any words.
            ValueError: If temperature is not greater than zero.
        """
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not self.unigrams:
            raise RuntimeError("the model must be trained before prediction")

        candidates: Mapping[str, int] | None = None

        if first_word is not None and second_word is not None:
            candidates = self.trigrams.get((first_word, second_word))

        latest_word = second_word if second_word is not None else first_word
        if candidates is None and latest_word is not None:
            candidates = self.bigrams.get(latest_word)

        if candidates is None:
            candidates = self.unigrams

        return self._sample(candidates, temperature)

    @staticmethod
    def _sample(counts: Mapping[str, int], temperature: float) -> str:
        """Sample a word after adjusting frequency weights by temperature."""
        words = list(counts)
        weights = [counts[word] ** (1.0 / temperature) for word in words]
        return random.choices(words, weights=weights, k=1)[0]
