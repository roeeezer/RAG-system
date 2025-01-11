from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from typing import List, Tuple, Set
import re

class HebrewSynonymExpander:
    def __init__(self, model_name="onlplab/alephbert-base", top_k=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Templates that are likely to produce synonyms
        # Each template should have exactly one MASK token and one TARGET placeholder
        self.templates = [
            "TARGET זה כמו MASK",  # "[word] is like [mask]"
            "TARGET או במילים אחרות MASK",  # "[word] or in other words [mask]"
            "TARGET דומה ל MASK",  # "[word] is similar to [mask]"
            "MASK זה מילה נרדפת ל TARGET",  # "[mask] is a synonym for [word]"
            "TARGET הוא MASK",  # "[word] is [mask]" (for nouns/adjectives)
        ]

    def get_predictions(self, text: str) -> List[Tuple[str, float]]:
        """Get model predictions for a masked position."""
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        # Find the masked position
        mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
        predicted_tokens = predictions[0, mask_position].squeeze()
        
        # Get top predictions
        top_tokens = torch.topk(predicted_tokens, self.top_k * 3)  # Get more candidates for filtering
        
        results = []
        for token_id, score in zip(top_tokens.indices, top_tokens.values):
            word = self.tokenizer.decode([token_id]).strip()
            prob = torch.softmax(score, dim=0).item()
            results.append((word, prob))
            
        return results

    def filter_candidates(self, candidates: List[Tuple[str, float]], original_word: str) -> List[Tuple[str, float]]:
        """Filter and clean candidate words."""
        filtered = []
        seen_words = set()
        
        for word, score in candidates:
            # Clean the word
            word = re.sub(r'[.,!?״׳]', '', word)  # Remove punctuation
            word = word.strip()
            
            # Apply filters
            if (len(word) > 1 and  # Not too short
                not word.startswith('##') and  # Not a subword
                word != original_word and  # Not the same as original
                word not in seen_words and  # Not duplicate
                not any(c.isdigit() for c in word) and  # No numbers
                all(c.isalpha() or c in 'םןץףך' for c in word)):  # Only Hebrew letters
                
                filtered.append((word, score))
                seen_words.add(word)
                
        return filtered[:self.top_k]

    def get_synonyms(self, word: str) -> List[Tuple[str, float]]:
        """Get synonyms for a word using multiple templates."""
        all_predictions = []
        
        for template in self.templates:
            # Replace placeholders with actual word and mask token
            text = template.replace('TARGET', word).replace('MASK', self.tokenizer.mask_token)
            predictions = self.get_predictions(text)
            all_predictions.extend(predictions)
        
        # Combine predictions from all templates
        word_scores = {}
        for word, score in all_predictions:
            if word in word_scores:
                word_scores[word] = max(word_scores[word], score)  # Keep highest score
            else:
                word_scores[word] = score
        
        # Convert back to list and filter
        candidates = [(word, score) for word, score in word_scores.items()]
        filtered_synonyms = self.filter_candidates(candidates, word)
        
        return filtered_synonyms

    def expand_query(self, query: str) -> str:
        """Expand a Hebrew search query with synonyms."""
        words = query.split()
        expanded_words = set(words)  # Start with original words
        
        for word in words:
            synonyms = self.get_synonyms(word)
            for synonym, _ in synonyms:
                expanded_words.add(synonym)
        
        return ' '.join(expanded_words)

def demo_usage():
    # Example usage
    expander = HebrewSynonymExpander(top_k=3)
    
    # Example queries
    queries = [
        "ספר מעניין",
        "בית גדול",
        "ללכת מהר",
    ]
    
    for query in queries:
        print(f"\nOriginal query: {query}")
        expanded = expander.expand_query(query)
        print(f"Expanded query: {expanded}")
        
        # Show synonyms for each word
        print("\nSynonyms for each term:")
        for word in query.split():
            synonyms = expander.get_synonyms(word)
            print(f"{word}:", ', '.join(f"{w} ({p:.3f})" for w, p in synonyms))

if __name__ == "__main__":
    pass  # Run demo_usage() to see the example
