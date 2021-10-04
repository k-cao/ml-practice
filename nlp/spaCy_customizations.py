import spacy
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab

# vocab: storage container - vocab data
#   ex. lexical types such as adjective, verb, noun
# tokens list: add to Doc obj
doc = Doc(Vocab(), words=[u'Hi', u'there'])
print(doc)
# Hi there

nlp = spacy.load('en_core_web_sm')

# Retrieve left syntactic dependencies & syntactic children of "apple"
# both are the same in this case
doc = nlp(u'I want a green apple')
[print(w) for w in doc[4].lefts]
[print(w) for w in doc[4].children]
# a
# green

# Retrieve right syntactic children of "want"
[print(w) for w in doc[1].rights]
# apple

# Separate into sentences
doc = nlp(u'A severe storm hit the city. It started to snow.')
for sent in doc.sents:
    print([sent[i] for i in range(len(sent))])
# [A, severe, storm, hit, the, city, .]
# [It, started, to, snow, heavily, .]

# Identify 1st word in a sentence
for i, sent in enumerate(doc.sents):
    if i == 1 and sent[0].pos_ == 'PRON':
        print('Second sentence begins with a pronoun')
# Second sentence begins with a pronoun

# Identify latter words in the sentence
numSentEndingInVerb = 0
for sent in doc.sents:
    # punctunation is the last elem, so take 2nd last
    if sent[len(sent) - 2].pos_ == 'VERB':
        numSentEndingInVerb += 1
print('Number of sentences ending with verb is %d' % numSentEndingInVerb)
# Number of sentences ending with verb is 1

# Noun Chunk: phrase w/ noun as its head
doc = nlp(u'A noun chunk is a phrase that has a noun as its head.')
print([chunk for chunk in doc.noun_chunks])
# [A noun chunk, a phrase, a noun, its head]

# can also compose noun chunks manually
chunks_list = []
chunk = None

for token in doc:
    if token.pos_=='NOUN':
        chunk = ''

        for w in token.children:
            if w.pos_ == 'DET' or w.pos_ == 'ADJ':
                chunk = chunk + w.text + ' '

        chunk += token.text
        chunks_list.append(chunk)
# [A noun chunk, a phrase, a noun, its head]

# Span: a slice from a Doc obj
doc = nlp(u'The Golden Gate Bridge is the inspiration behind the Cisco logo.')
print([doc[i] for i in range(len(doc))])
# [The, Golden, Gate, Bridge, is, the, inspiration, behind, the, Cisco, logo, .]

with doc.retokenize() as retokenizer:
    span = doc[1:4] # Golden Gate Bridge
    retokenizer.merge(span)

print([doc[i] for i in range(len(doc))])
# [The, Golden Gate Bridge, is, the, inspiration, behind, the, Cisco, logo, .]
print([(token.text, token.lemma_, token.pos_, token.dep_) for token in doc]) 
# [('The', 'the', 'DET', 'det'), ('Golden Gate Bridge', 'Golden Gate Bridge', 'PROPN', 'nsubj'), ('is', 'be', 'AUX', 'ROOT'), ('the', 'the', 'DET', 'det'), ('inspiration', 'inspiration', 'NOUN', 'attr'), ('behind', 'behind', 'ADP', 'prep'), ('the', 'the', 'DET', 'det'), ('Cisco', 'Cisco', 'PROPN', 'compound'), ('logo', 'logo', 'NOUN', 'pobj'), ('.', '.', 'PUNCT', 'punct')]

# available pipeline components
print(nlp.pipe_names)
# ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']

# can disable pipeline components
nlp = spacy.load('en_core_web_sm', disable=['parser'])
doc = nlp(u'I want a green apple.')
# disabling parser => no dependency labels 
print([(token.text, token.pos_, token.dep_) for token in doc])
# [('I', 'PRON', ''), ('want', 'VERB', ''), ('a', 'DET', ''), ('green', 'ADJ', ''), ('apple', 'NOUN', ''), ('.', 'PUNCT', '')]

nlp = spacy.load('en_core_web_sm')

# find location in system
from spacy import util
model_data_path = util.get_package_path('en_core_web_sm')
print(model_data_path)
# it's in /lib of virtualenv workspace

doc = nlp(u'I need a taxi to Festy and then a plane to Berlin.')
print('Entities: {}'.format([(ent.text, ent.label_) for ent in doc.ents]))
# Entities: [('Berlin', 'GPE')] -- nothing for Festy

LABEL = 'DISTRICT'
TRAIN_DATA = [
    ('We need to deliver it to Festy.', {
        'entities': [(25, 30, LABEL)]
    }),
    ('I like red oranges', {
        'entities': []
    })
]

ner = nlp.get_pipe('ner')
ner.add_label(LABEL)

nlp.disable_pipes('tagger')
nlp.disable_pipes('parser')

optimizer = ner.create_optimizer()

import random
from spacy.training.example import Example

for i in range(25):
    for batch in spacy.util.minibatch(TRAIN_DATA, size=1):
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            nlp.update([Example.from_dict(doc, annotations)])

doc = nlp('We need to deliver it to Festy.',)
print([(ent.text, ent.label_) for ent in doc.ents])
# [('Festy', 'DISTRICT')]

