#
# A simple endpoint that can receive documents from an external source, mark them up and return them.  This can be useful
# for hooking in callback functions during indexing to do smarter things like classification
#
from flask import (
    Blueprint, request, abort, current_app, jsonify
)
import fasttext
from nltk.stem import SnowballStemmer
import json

bp = Blueprint('documents', __name__, url_prefix='/documents')

# Take in a JSON document and return a JSON document
@bp.route('/annotate', methods=['POST'])
def annotate():
    if request.mimetype == 'application/json':
        the_doc = request.get_json()
        response = {}
        cat_model = current_app.config.get("cat_model", None) # see if we have a category model
        syns_model = current_app.config.get("syns_model", None) # see if we have a synonyms/analogies model
        # We have a map of fields to annotate.  Do POS, NER on each of them
        sku = the_doc["sku"]
        threshold = 0.9
        for item in the_doc:
            the_text = the_doc[item]
            if the_text is not None and the_text.find("%{") == -1:
                if item == "name":
                    if syns_model is not None:
                        stemmer = SnowballStemmer("english")
                        tokens = nltk.word_tokenize(product_name)
                        new_words = [w.lower() for w in tokens]
                        stem_words = [stemmer.stem(w) for w in new_words]
                        text_transformed = " ".join(stem_words)  
                        for item in syns_model.get_nearest_neighbors(text_transformed):
                            if "name_synonyms" not in response:
                                response["name_synonyms"] = []
                            if(item[0] > threshold):
                                response["name_synonyms"].append(item[1])
                        
        return jsonify(response)
    abort(415)
