from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from typing import List, Tuple
import numpy as np

class HebrewSemanticExpander:
    def __init__(self, model_name="onlplab/alephbert-base", top_k=5):
        """
        Initialize with a Hebrew transformer model.
        Default is AlephBERT, which is pre-trained on Hebrew text.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_similar_words(self, word: str) -> List[Tuple[str, float]]:
        """
        Get similar words using masked language modeling.
        Returns list of (word, probability) tuples.
        """
        # Create masked input
        masked_text = f"{word} {self.tokenizer.mask_token}"
        inputs = self.tokenizer(masked_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # Get the predicted tokens for the masked position
        mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
        predicted_tokens = predictions[0, mask_position].squeeze()
        
        # Get top k predictions
        top_tokens = torch.topk(predicted_tokens, self.top_k * 2)  # Get more candidates to filter later
        
        # Convert to words and probabilities
        similar_words = []
        for token_id, score in zip(top_tokens.indices, top_tokens.values):
            word = self.tokenizer.decode([token_id]).strip()
            prob = torch.softmax(score, dim=0).item()
            
            # Filter out subwords and very short words
            if len(word) > 1 and not word.startswith('##'):
                similar_words.append((word, prob))
        
        return similar_words[:self.top_k]

    def expand_query(self, query: str) -> str:
        """
        Expand a Hebrew search query using masked language modeling.
        Returns space-separated expanded query.
        """
        words = query.split()
        expanded_words = set(words)  # Start with original words
        
        for word in words:
            # Get similar words
            similar_words = self.get_similar_words(word)
            
            # Add similar words to expanded set
            for similar_word, _ in similar_words:
                expanded_words.add(similar_word)
        
        return ' '.join(expanded_words)

def demo_usage():
    # Example usage
    expander = HebrewSemanticExpander(top_k=3)
    
    # Example queries
    queries = [
        "יש מצב שניתן להשאיר קטין בבית סוהר לא מופרד למשך יום?",
        "מה צריך להופיע בחוזה העסקה של עובד זר?",
    ]
    
    for query in queries:
        print(f"\nOriginal query: {query}")
        expanded = expander.expand_query(query)
        print(f"Expanded query: {expanded}")
        
        # Show similar words for each word in query
        print("\nSimilar words for each term:")
        for word in query.split():
            similar_words = expander.get_similar_words(word)
            print(f"{word}:", ', '.join(f"{w} ({p:.3f})" for w, p in similar_words))

if __name__ == "__main__":
    demo_usage()
