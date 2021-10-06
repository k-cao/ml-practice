import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(u"The firm earned $1.5 million in 2017.")

# coarse-grained
[print(token.text, token.pos_, spacy.explain(token.pos_)) for token in doc]
# The DET determiner
# firm NOUN noun
# earned VERB verb
# $ SYM symbol
# 1.5 NUM numeral
# million NUM numeral
# in ADP adposition
# 2017 NUM numeral
# . PUNCT punctuation

print()

# fine-grained
[print(token.text, token.tag_, spacy.explain(token.tag_)) for token in doc]
# The DT determiner
# firm NN noun, singular or mass
# earned VBD verb, past tense
# $ $ symbol, currency
# 1.5 CD cardinal number
# million CD cardinal number
# in IN conjunction, subordinating or preposition
# 2017 CD cardinal number
# . . punctuation mark, sentence closer

print()

phrase = ''
i = 0
for token in doc:
    if token.tag_ == '$':
        phrase = token.text
        i = token.i + 1

        while doc[i].tag_ == 'CD':
            phrase += doc[i].text + ' '
            i += 1

        break
phrase = phrase[:-1]
print(phrase)
# $1.5 million

# confirmatory sentence: Can you really promise it is worth my time?
doc = nlp(u'I can promise it is worth your time.')
print([(token.text, token.pos_, token.tag_) for token in doc])
# [('I', 'PRON', 'PRP'), ('can', 'AUX', 'MD'), ('promise', 'VERB', 'VB'), 
# ('it', 'PRON', 'PRP'), ('is', 'AUX', 'VBZ'), ('worth', 'ADJ', 'JJ'), 
# ('your', 'PRON', 'PRP$'), ('time', 'NOUN', 'NN'), ('.', 'PUNCT', '.')]

print()

sent = ''
for i, token in enumerate(doc):
    if token.tag_ == 'PRP' and doc[i+1].tag_ == 'MD' and doc[i+2].tag_ == 'VB':
        sent = doc[i+1].text.capitalize() + ' ' + doc[i].text
        sent = sent + ' ' + doc[i+2:].text
        break
print(sent)
# Can I promise it is worth your time.

doc = nlp(sent)
for i, token in enumerate(doc):
    if token.tag_ == 'PRP' and token.text == 'I':
        sent = doc[:i].text + ' you ' + doc[i+1:].text
        break
print(sent)
# Can you promise it is worth your time.

doc = nlp(sent)
for i, token in enumerate(doc):
    if token.tag_ == 'PRP$' and token.text == 'your':
        sent = doc[:i].text + ' my ' + doc[i+1:].text
        break
print(sent)
# Can you promise it is worth my time.

doc = nlp(sent)
for i,token in enumerate(doc):
    if token.tag_ == 'VB':
        sent = doc[:i].text + ' really ' + doc[i:].text
        break
print(sent)
# Can you really promise it is worth my time.

doc = nlp(sent)
sent = doc[:len(doc)-1].text + '?'
print(sent)
# Can you really promise it is worth my time?
