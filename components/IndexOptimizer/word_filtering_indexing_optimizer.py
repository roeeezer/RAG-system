from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface

words_to_filter_by_category = [
    ('ה', 'article'),
    ('ו', 'conjunction'),
    ('או', 'conjunction'),
    ('אבל', 'conjunction'),
    ('כי', 'conjunction'),
    ('ב', 'preposition'),
    ('ל', 'preposition'),
    ('כ', 'preposition'),
    ('על', 'preposition'),
    ('עם', 'preposition'),
    ('לפני', 'preposition'),
    ('אחרי', 'preposition'),
    ('מתחת', 'preposition'),
    ('מעל', 'preposition'),
    ('בין', 'preposition'),
    ('אצל', 'preposition'),
    ('אני', 'pronoun'),
    ('אתה', 'pronoun'),
    ('את', 'pronoun'),
    ('הוא', 'pronoun'),
    ('היא', 'pronoun'),
    ('אנחנו', 'pronoun'),
    ('אתם', 'pronoun'),
    ('אתן', 'pronoun'),
    ('הם', 'pronoun'),
    ('הן', 'pronoun'),
    ('היה', 'auxiliary verb'),
    ('היו', 'auxiliary verb'),
    ('להיות', 'auxiliary verb'),
    ('יש', 'auxiliary verb'),
    ('אין', 'auxiliary verb'),
    ('מה', 'other'),
    ('מי', 'other'),
    ('זה', 'other'),
    ('זאת', 'other'),
    ('כל', 'other'),
    ('שום', 'other'),
    ('כן', 'other'),
    ('לא', 'other'),
    ('מאוד', 'other'),
    ('רק', 'other'),
    ('גם', 'other'),
    ('כבר', 'other'),
    ('אם', 'other'),
    ('שם', 'other'),
    ('כאן', 'other'),
    ('עכשיו', 'other'),
    ('בבקשה', 'other'),
]


class WordFilteringIndexingOptimizer(IndexingTextOptimizerInterface):

    def __init__(self):
        self.words_to_filter = [word for word, _ in words_to_filter_by_category]

    def optimize_query(self, text: str) -> str:
        return self.optimize_text(text)

    def optimize_document(self, text: str) -> str:
        return self.optimize_text(text)

    def optimize_text(self, text: str) -> str:
        text = text.split()
        text = [word for word in text if word not in self.words_to_filter]
        return ' '.join(text)