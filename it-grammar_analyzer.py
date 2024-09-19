from flask import Flask, request, jsonify
import spacy
from spacy import displacy
from textblob import TextBlob

# Initialize Flask app
app = Flask(__name__)

# Load the Italian language model
nlp = spacy.load("it_core_news_sm")

# Additional Models and Libraries to Include Later:
# Example: Sentiment Analysis, Named Entity Recognition (NER)

def analyze_sentence(sentence):
    doc = nlp(sentence)
    analysis = []
    
    # Syntax and Dependency Parsing
    for token in doc:
        analysis.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "role": get_role(token),
            "is_stop": token.is_stop  # Check if the word is a stop word
        })

    # Named Entity Recognition (NER)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    return {"tokens": analysis, "entities": entities}

def get_role(token):
    """Extended role determination with dependency labels."""
    if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
        return "Subject"
    elif token.dep_ == 'ROOT' and token.pos_ == 'VERB':
        return "Predicate"
    elif token.dep_ in {'obj', 'dobj'}:
        return "Object"
    elif token.dep_ == 'iobj':
        return "Indirect Object"
    elif token.dep_ == 'advmod':
        return "Adverb"
    elif token.dep_ == 'amod':
        return "Adjective"
    elif token.dep_ == 'det':
        return "Determiner"
    elif token.dep_ == 'prep':
        return "Preposition"
    # Extend with more detailed grammatical roles
    else:
        return "Other"

def correct_sentence(sentence):
    """Grammar correction using TextBlob or other libraries."""
    blob = TextBlob(sentence)
    return str(blob.correct())

def analyze_sentiment(sentence):
    """Perform sentiment analysis using TextBlob."""
    blob = TextBlob(sentence)
    return blob.sentiment

def conjugate_word(word):
    """Conjugate a verb (extend with Italian conjugation libraries)."""
    # Placeholder: Needs integration with an Italian conjugation library
    return {"original": word, "conjugated": f"{word} (conjugated form)"}

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for analyzing a sentence."""
    try:
        data = request.get_json()
        sentence = data.get('text', '')

        if not sentence:
            return jsonify({"error": "No text provided"}), 400

        # Perform the analysis
        result = analyze_sentence(sentence)
        result["correction"] = correct_sentence(sentence)  # Grammar correction
        result["sentiment"] = analyze_sentiment(sentence)  # Sentiment analysis

        return jsonify(result)

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/conjugate', methods=['POST'])
def conjugate():
    """API endpoint to conjugate a word."""
    try:
        data = request.get_json()
        word = data.get('word', '')

        if not word:
            return jsonify({"error": "No word provided"}), 400

        conjugated = conjugate_word(word)
        return jsonify(conjugated)

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/display', methods=['POST'])
def display():
    """API endpoint for visualizing sentence dependency."""
    try:
        data = request.get_json()
        sentence = data.get('text', '')

        if not sentence:
            return jsonify({"error": "No text provided"}), 400

        doc = nlp(sentence)
        svg = displacy.render(doc, style="dep", jupyter=False)
        return jsonify({"svg": svg})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/tool', methods=['GET'])
def tool():
    return "Tool is working"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
