from flask import Flask, request, jsonify, render_template
import aiml
import re
import random
import string
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# membuat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

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
    return max(candidates(word), key=P)

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

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

GENERATE_RESPONSE = ['Maaf, saya tidak mengerti pertanyaan Anda.', 
                     'Sayangnya, saya tidak memiliki informasi yang relevan untuk pertanyaan Anda, bisakah saya membantu Anda dengan topik lain?',
                     'Maaf, saya belum belajar tentang topik tersebut, apakah ada pertanyaan lain yang bisa saya bantu jawab?',
                     'Mohon maaf, saya belum mengetahui informasi yang relevan terkait pertanyaan Anda. Bisakah saya membantu Anda dengan topik lain?'
                     ]

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    user_input = data.get("user_input")
    if not user_input:
        return jsonify({"response": "Tolong masukkan input yang valid."}), 400
    try:
        preprocessed_input = preprocess(user_input)
    except Exception as e:
        return jsonify({"response": "Terjadi kesalahan dalam pre-processing input."}), 500
    try:
        response = kernel.respond(preprocessed_input)
    except Exception as e:
        return jsonify({"response": "Terjadi kesalahan dalam memproses input."}), 500
    if not response or response == "":
        response = (random.choice(GENERATE_RESPONSE))
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
