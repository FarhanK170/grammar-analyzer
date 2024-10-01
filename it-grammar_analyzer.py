from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from spacy import displacy
import language_tool_python
from textblob import TextBlob

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the Italian language model from SpaCy
nlp = spacy.load("it_core_news_lg")  # Using the large Italian model for better accuracy

# Initialize LanguageTool for Italian grammar correction
tool = language_tool_python.LanguageToolPublicAPI('it')

def analyze_sentence(sentence):
    # Analyze the sentence using SpaCy
    doc = nlp(sentence)
    analysis = []

    # Detailed analysis with extended grammatical roles
    for token in doc:
        analysis.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "role": get_detailed_role(token),
            "logical_analysis": get_logical_complement(token),
            "is_stop": token.is_stop,
            "morph": token.morph.to_dict()  # Include morphological details
        })

    # Named Entity Recognition (NER)
    entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]

    return {"tokens": analysis, "entities": entities, "logical_analysis_summary": summarize_logical_analysis(doc)}

def get_detailed_role(token):
    """Extended role determination with more detailed grammatical explanations."""
    role_map = {
        "nsubj": "Soggetto",
        "ROOT": "Predicato Verbale",
        "obj": "Complemento Oggetto",
        "iobj": "Complemento di Termine",
        "advmod": "Avverbio Modificatore",
        "amod": "Aggettivo Modificatore",
        "det": "Determinante",
        "prep": "Preposizione",
        "aux": "Verbo Ausiliare",
        "cc": "Congiunzione Coordinante",
        "mark": "Congiunzione Subordinante",
        "obl": "Complemento Indiretto",
        "ccomp": "Proposizione Oggettiva",
        "xcomp": "Complemento Predicativo dell'Oggetto",
        "punct": "Punteggiatura"
    }

    # Return more detailed descriptions based on the token's dependency label
    return role_map.get(token.dep_, "Altro")

def get_logical_complement(token):
    """Map the token dependency to detailed logical complement categories."""
    logical_complements = {
        "obj": "Complemento Oggetto Diretto",
        "obl": "Complemento Circostanziale",
        "iobj": "Complemento Indiretto",
        "xcomp": "Complemento Predicativo dell'Oggetto",
        "ccomp": "Proposizione Oggettiva",
        "nsubj": "Soggetto Logico",
        "advmod": "Complemento di Modo o Maniera",
        "prep": "Preposizione Logica",
        "pobj": "Oggetto Preposizionale"
    }

    return logical_complements.get(token.dep_, "Altro Complemento")

def correct_sentence(sentence):
    """Grammar correction using LanguageTool for Italian."""
    matches = tool.check(sentence)
    corrected_sentence = language_tool_python.utils.correct(sentence, matches)
    return corrected_sentence

def analyze_sentiment(sentence):
    """Perform sentiment analysis using TextBlob with Italian-to-English translation."""
    blob = TextBlob(sentence)
    translated_sentence = str(blob.translate(to='en'))
    translated_blob = TextBlob(translated_sentence)
    return translated_blob.sentiment

def conjugate_word(word):
    """Conjugate a verb (extend with Italian conjugation libraries)."""
    # Placeholder: Needs integration with an Italian conjugation library
    return {"original": word, "conjugated": f"{word} (forma coniugata)"}

def summarize_logical_analysis(doc):
    """Summarize the logical analysis into comprehensive categories."""
    summary = {
        "Soggetti": [token.text for token in doc if token.dep_ == 'nsubj'],
        "Predicati Verbali": [token.text for token in doc if token.dep_ == 'ROOT'],
        "Complementi Oggetto": [token.text for token in doc if token.dep_ == 'obj'],
        "Complementi Indiretti": [token.text for token in doc if token.dep_ == 'obl'],
        "Preposizioni": [token.text for token in doc if token.dep_ == 'prep']
    }
    return summary

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

@app.route('/display', methods=['POST'])
def display():
    """API endpoint for visualizing sentence dependency."""
    try:
        data = request.get_json()
        sentence = data.get('text', '')

        if not sentence:
            return jsonify({"error": "No text provided"}), 400

        doc = nlp(sentence)
        svg = displacy.render(doc, style="dep", jupyter=False, options={"compact": True, "color": "blue"})
        return jsonify({"svg": svg})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/tool', methods=['GET'])
def tool():
    return "Tool is working"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
