from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Tuple, Set
import re

class HebrewSynonymExpander:
    def __init__(self, model_name="onlplab/alephbert-base", top_k=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Add MLM head
        self.mlm_head = torch.nn.Linear(
            self.model.config.hidden_size, 
            self.model.config.vocab_size
        ).to(self.device)
        
        self.model.eval()
        self.mlm_head.eval()

        # Templates for synonym generation
        self.templates = [
            "TARGET זה כמו MASK",
            "TARGET או במילים אחרות MASK",
            "TARGET דומה ל MASK",
            "MASK זה מילה נרדפת ל TARGET",
            "TARGET הוא MASK",
        ]

    def get_predictions(self, text: str) -> List[Tuple[str, float]]:
        """Get model predictions for a masked position."""
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get hidden states from base model
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            # Find mask position and get its hidden state
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
            mask_hidden_state = hidden_states[0, mask_position].squeeze()
            
            # Get predictions using MLM head
            logits = self.mlm_head(mask_hidden_state)
            
            # Get top predictions
            top_tokens = torch.topk(logits, self.top_k * 3)
            
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
            word = re.sub(r'[.,!?״׳]', '', word)
            word = word.strip()
            
            # Apply filters
            if (len(word) > 1 and
                not word.startswith('##') and
                word != original_word and
                word not in seen_words and
                not any(c.isdigit() for c in word) and
                all(c.isalpha() or c in 'םןץףך' for c in word)):
                
                filtered.append((word, score))
                seen_words.add(word)
                
        return filtered[:self.top_k]

    def get_synonyms(self, word: str) -> List[Tuple[str, float]]:
        """Get synonyms for a word using multiple templates."""
        all_predictions = []
        
        for template in self.templates:
            text = template.replace('TARGET', word).replace('MASK', self.tokenizer.mask_token)
            predictions = self.get_predictions(text)
            all_predictions.extend(predictions)
        
        # Combine predictions from all templates
        word_scores = {}
        for word, score in all_predictions:
            if word in word_scores:
                word_scores[word] = max(word_scores[word], score)
            else:
                word_scores[word] = score
        
        # Convert back to list and filter
        candidates = [(word, score) for word, score in word_scores.items()]
        filtered_synonyms = self.filter_candidates(candidates, word)
        
        return filtered_synonyms

    def expand_query(self, query: str) -> str:
        """Expand a Hebrew search query with synonyms."""
        words = query.split()
        expanded_words = set(words)
        
        for word in words:
            synonyms = self.get_synonyms(word)
            for synonym, _ in synonyms:
                expanded_words.add(synonym)
        
        return ' '.join(expanded_words)

def demo_usage():
    expander = HebrewSynonymExpander(top_k=3)
    
    queries = [
        "ספר מעניין",
        "בית גדול",
        "ללכת מהר",
    ]
    
    for query in queries:
        print(f"\nOriginal query: {query}")
        expanded = expander.expand_query(query)
        print(f"Expanded query: {expanded}")
        
        print("\nSynonyms for each term:")
        for word in query.split():
            synonyms = expander.get_synonyms(word)
            print(f"{word}:", ', '.join(f"{w} ({p:.3f})" for w, p in synonyms))

if __name__ == "__main__":
    demo_usage()
