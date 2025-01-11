from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface
from typing import List


class PrefixSuffixSplitterOptimizer(IndexingTextOptimizerInterface):

    def __init__(self):
        self.prefixes_to_split = [
            "ה",
            "ו",
            "ב",
            "ל",
            "ש",
            "כ",
            "מ",            
        ]
        self.suffixes_to_split = [
            "ים",
            "ות",
            "י",
            "ה",
            "ו",
        ]

    def optimize_queries(self, lst_text: List[str]) -> List[str]:
        res = []
        for text in lst_text:
            res.append(self.optimize_text(text))
        return res

    def optimize_documents(self, lst_text: List[str]) -> List[str]:
        return self.optimize_queries(lst_text)

    def optimize_text(self, text: str) -> str:
        text = text.split()
        new_text = []
        for word in text:
            self.append_word(new_text, word)
            self.split_prefixes(new_text, word)
            self.split_suffixes(new_text, word)
        return " ".join(new_text)

    def split_suffixes(self, new_text, splitted_word):
        while (len(splitted_word) >= 1 and splitted_word[-1] in self.suffixes_to_split) or \
            (len(splitted_word) >= 2 and splitted_word[-2] in self.suffixes_to_split) :
            if len(splitted_word) >= 2 and splitted_word[-2] in self.suffixes_to_split:
                splitted_word = splitted_word[:-2]
            else:
                splitted_word = splitted_word[:-1]
            self.append_word(new_text, splitted_word)

    def split_prefixes(self, new_text, splitted_word):
        while len(splitted_word) >= 1 and splitted_word[0] in self.prefixes_to_split:
            splitted_word = splitted_word[1:]
            self.append_word(new_text, splitted_word)
            self.split_suffixes(new_text, splitted_word)
    
    def append_word(self, new_text: list[str], word: str):
        if len(word) <= 1:
            return
        
        # Check if last letter needs conversion
        last_letter = word[-1]
        if last_letter in "כמנפצ":
            # Only convert to list if needed
            word_chars = list(word)
            word_chars[-1] = self.convert_to_ot_sofit(last_letter)
            word = ''.join(word_chars)
        
        new_text.append(word)

    def convert_to_ot_sofit(self, letter):
        if letter == "כ":
            return "ך"
        if letter == "מ":
            return "ם"
        if letter == "נ":
            return "ן"
        if letter == "פ":
            return "ף"
        if letter == "צ":
            return "ץ"
        return letter
