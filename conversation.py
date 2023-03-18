import aiml
import re
import string
import spacy
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# membuat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

nlp = spacy.load("xx_ent_wiki_sm")

def ner(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

# membuat kernal dan memuat file aiml
kernel = aiml.Kernel()
try:
    kernel.learn("startup.xml")
    kernel.respond("load aiml a")
except Exception as e:
    print("Error: ", e)
    # jika terjadi kesalahan, keluar dari program
    exit()

# membaca big_corpus
with open('data/kata-dasar.txt', 'r') as file:
    big_corpus = ""
    for line in file:
        big_corpus += line

# kamus untuk spelling correction
def words(text):
    return re.findall(r'\w+', text.lower())

WORDS = Counter(words(big_corpus))

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    try:
        return max(candidates(word), key=P)
    except ValueError:
        return word
        
def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in known(edits1(e1)))

def update_knowledge(text):
    entities = ner(text)
    for entity in entities:
        pattern = entity[0] + ' ' + entity[1]
        template = 'Tell me more about ' + entity[0] + '.'
        category = '<category><pattern>{}</pattern><template>{}</template></category>'.format(pattern, template)
        kernel.learn(category)

# function untuk pre-processing
def preprocess(text):
    try:
        # case folding
        text = text.lower()

        # menghilangkan nomor and tanda baca
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))

        # tokenizing
        words = text.split()

        # spelling correction
        corrected_words = []
        for word in words:
            corrected_word = correction(word)
            corrected_words.append(corrected_word)

        # stemming
        stemmed_words = []
        for word in corrected_words:
            stemmed_word = stemmer.stem(word)
            stemmed_words.append(stemmed_word)

        # filtering stopwords
        with open('data/daftar-stopword.txt', 'r') as file:
            stopwords = file.read().splitlines()
        filtered_words = [word for word in stemmed_words if word not in stopwords]

        # join words back to sentence
        preprocessed_text = ' '.join(filtered_words)

        return preprocessed_text
    except FileNotFoundError as e:
        # jika file tidak ditemukan, tampilkan pesan error yang lebih jelas
        print("Error: File tidak ditemukan -", e.filename)
        return text
    except AttributeError as e:
        # jika terjadi kesalahan atribut, tampilkan pesan error yang lebih jelas
        print("Error: AttributeError -", e)
        return text

# function for chatting
def chat(text):
    preprocessed_text = preprocess(text)
    response = kernel.respond(preprocessed_text)
    return response
