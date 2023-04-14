import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
from nltk import ne_chunk

## In this question, you'll get a little bit of practice using the NLTK chunker to extract
## structured knowledge.


## This function will take a string representing a sentence and give us back a list of tagged tokens.
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


example1 = 'Tens of thousands marched as protests and strikes against unpopular pension reforms gripped France again Tuesday, with police ramping up security after the government warned that radical demonstrators intended to destroy, to injure and to kill.'
example2 = "The United States and China are in talks to ease tensions over trade, according to officials from both countries."
example3 = "The CEO of Apple, Tim Cook, announced a new line of products at the company's annual conference in Cupertino."
example4 = "The United Nations warned of a humanitarian crisis in Yemen due to the ongoing war between government forces and Houthi rebels."
example5 = "China warns of retaliation after Australia cancels deal to build undersea communications cable."
### Use your favorite news source to find five more sentences to use as input.


## This piece of code builds a Regex parser to recognize noun phrases.
## It says that a noun phrase is 0 or 1 determiners (a, an, the), followed by 0 or more adjectives (JJ),
## followed by a noun.
## Run it on your examples. Does it get them right? Are there cases in which a noun phrase was not recognized?
## why do you think that is?

sentences = [example1, example2, example3, example4, example5]

for sent in sentences:
    print("Sentence:", sent)

    # Noun Phrase Chunking
    s = preprocess(sent)
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(s)
    print("Noun Phrase Chunking:", cs)

    # Named Entity Recognition
    ne_tree = ne_chunk(pos_tag(word_tokenize(sent)))
    print("Named Entity Recognition:")
    pprint(tree2conlltags(ne_tree))
    print("\n")

## NLTK also contains its own built-in tool for recognizing named entities.
## Note that, in the tree, enities are tagged with tags such as GPE (geo-political entity),
# Organization, or Person. https://www.nltk.org/book/ch07.html lists all of the types.


## Run the named entity chunker over your examples. What did it get right? What did it get wrong?

