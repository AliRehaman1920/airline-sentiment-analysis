""" lemmatizing, removing links, @usernames, punctuations, and domain specific words """

import spacy

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner']) # disabling parser and ner because they are not needed

# returns the text with lemmatized words

def clean_data(texts): # parameters: a column of text

    cleaned_texts = texts.str.replace(r'@\w+', '', regex=True) # to remove @names
    cleaned_texts = cleaned_texts.str.replace(r'https?\S+', '', regex=True) # to remove http | https
    cleaned_texts = cleaned_texts.str.replace(r'\s+', ' ', regex=True).str.strip() # to remove extra spaces which came because of previous

    docs = nlp.pipe(cleaned_texts.astype(str), batch_size = 50)
    return [
        ' '.join(
            token.lemma_.lower() for token in doc
            if not token.is_stop and token.is_alpha
        )
        for doc in docs]


# returns texts by removing words which are in domain_words list
def rm_domain_words(texts, domain_words):   # parameters: a column of text and a list of words astype(str)
    return texts.apply(
        lambda text: ' '.join(
            word for word in text.split()
            if word.lower() not in domain_words
            ))                              
    