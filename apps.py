from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

app = Flask(__name__)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Convert treebank tags to wordnet tags."""
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return None  # If not a recognized part of speech

def lemmatize_text(text):
    # Tokenize the input text
    words = word_tokenize(text)
    # Get part-of-speech tags
    pos_tags = pos_tag(words)
    # Lemmatize each word with its part of speech
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos) or 'n') for word, pos in pos_tags
    ]
    # Join the lemmatized words back into a single string
    return ' '.join(lemmatized_words)

@app.route('/', methods=['GET', 'POST'])
def index():
    lemmatized_text = ""  # Initialize lemmatized_text
    if request.method == 'POST':
        input_text = request.form['text']
        
        # Lemmatize the input text
        lemmatized_text = lemmatize_text(input_text)
               
    return render_template('index.html', lemmatized_text=lemmatized_text)

if __name__ == '__main__':
    app.run(debug=True)
