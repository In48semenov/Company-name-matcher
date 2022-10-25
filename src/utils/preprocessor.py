import unicodedata
from transliterate import translit


def preprocessing(text: str) -> str:
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    try:
        return translit(text, reversed=True)
    except:
        return text
