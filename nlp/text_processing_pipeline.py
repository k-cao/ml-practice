import spacy

nlp = spacy.load('en_core_web_sm')

sentence = u'I am flying to DC'

# === Tokenization ===
doc = nlp(sentence)
print([w.text for w in doc])
# ['I', 'am', 'flying', 'to', 'DC']

# === Lemmatization ===
print([(token.text, token.lemma_) for token in doc])
# [('I', 'I'), ('am', 'be'), ('flying', 'fly'), ('to', 'to'), ('DC', 'DC')]

nlp.get_pipe('attribute_ruler').add(
    [[{'TEXT': 'DC'}]],
    {'LEMMA': 'Washington'}
)
doc = nlp(sentence)    
print([(token.text, token.lemma_) for token in doc])
# [('I', 'I'), ('am', 'be'), ('flying', 'fly'), ('to', 'to'), ('DC', 'Washington')]

# ======

sentence = u'I have flown to Berlin. Now I am flying to DC.'

# === Tagging ===
doc = nlp(sentence)

#   === Part-of-Speech ===

# Fine-grained part-of-speech
# VBG: present progressive form verb
# VB:  base form verb
print([w.text for w in doc if w.tag_== 'VBG' or w.tag_== 'VB'])
# ['flying']

# Coarse-grained part-of-speech
# PROPN: proper noun
print([w.text for w in doc if w.pos_ == 'PROPN'])
# ['Berlin', 'DC']

print()

#   === Dependency Labels ===
[print(token.text, token.pos_, token.dep_) for token in doc]
# I PRON nsubj
# have AUX aux
# flown VERB ROOT
# to ADP prep
# Berlin PROPN pobj
# . PUNCT punct
# Now ADV advmod
# I PRON nsubj
# am AUX aux
# flying VERB ROOT
# to ADP prep
# DC PROPN pobj
# . PUNCT punct

print()

# === Syntactic Dependency Parsing === 

# find dependency arcs
[print(token.head.text, token.dep_, token.text) for token in doc]
# flown nsubj I
# flown aux have
# flown ROOT flown
# flown prep to
# to pobj Berlin
# flown punct .
# flying advmod Now
# flying nsubj I
# flying aux am
# flying ROOT flying
# flying prep to
# to pobj DC
# flying punct .

print()

# find words assigned to dependency labels
#   ROOT: token whose head is itself
#   pobj: obj of preposition
for sent in doc.sents:
    print([w.text for w in sent if w.dep_ == 'ROOT' or w.dep_ == 'pobj'])
# ['flown', 'Berlin']
# ['flying', 'DC']

# === Named Entity Recognition ===
# find the entities, which are all geopolitical entities
for token in doc:
    if token.ent_type != 0:
        print(token.text, token.ent_type_)
# Berlin GPE
# DC GPE