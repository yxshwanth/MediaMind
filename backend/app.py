import os
import uuid
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import tempfile

# Firebase Admin SDK for Storage
import firebase_admin
from firebase_admin import credentials, storage

# ElasticSearch client
from elasticsearch import Elasticsearch

# For image and video processing
import torch
import clip  # OpenAI's CLIP package
from PIL import Image
import cv2

# For audio tagging using Musicnn
from musicnn.tagger import top_tags
import librosa

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from transformers import pipeline

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords and tokenizer if not already installed
nltk.download("stopwords")
nltk.download("punkt")

app = Flask(__name__)
CORS(app)

# Firebase configuration
FIREBASE_CRED_FILE = os.environ.get('FIREBASE_CRED_FILE', 'firebase_credentials.json')
FIREBASE_BUCKET = os.environ.get('FIREBASE_BUCKET', '<your-bucket-name>')
cred = credentials.Certificate(FIREBASE_CRED_FILE)
firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_BUCKET})
bucket = storage.bucket()

# Elasticsearch configuration
ELASTIC_HOST = os.environ.get('ELASTIC_HOST', 'localhost:9200')
ELASTIC_USER = os.environ.get('ELASTIC_USER')
ELASTIC_PASS = os.environ.get('ELASTIC_PASS')
if ELASTIC_USER and ELASTIC_PASS:
    es = Elasticsearch([ELASTIC_HOST], basic_auth=(ELASTIC_USER, ELASTIC_PASS))
else:
    es = Elasticsearch([ELASTIC_HOST])
INDEX_NAME = "media_files"

# Create index mapping
mapping = {
    "mappings": {
        "properties": {
            "tags": {"type": "keyword"},
            "filename": {"type": "text"},
            "url": {"type": "keyword"},
            "type": {"type": "keyword"},
            "format": {"type": "keyword"},
            "metadata": {"type": "object"}
        }
    }
}

# Create the index only if it doesn't exist
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created Elasticsearch index '{INDEX_NAME}'")
else:
    print(f"Elasticsearch index '{INDEX_NAME}' already exists.")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Candidate labels for tagging
CANDIDATE_LABELS = [
    "landscape", "cityscape", "portrait", "animal", "food", "indoor", "outdoor",
    "nature", "people", "sunset", "beach", "mountains", "ocean", "party", "sport",
    "text", "document", "car", "selfie", "night", "snow", "rain", "white", "mercedes",
    "trees", "dogs", "TedX", "architecture", "bridge", "river", "forest", "park",
    "garden", "landmark", "sky", "clouds", "waterfall", "desert", "canyon", "lake",
    "island", "sunrise", "dusk", "twilight", "urban", "suburban", "rural", "road",
    "highway", "street", "alley", "market", "festival", "concert", "performance",
    "crowd", "exhibition", "museum", "gallery", "painting", "sculpture", "vintage",
    "modern", "classic", "abstract", "minimal", "colorful", "monochrome", "black",
    "blue", "red", "green", "yellow", "purple", "orange", "pink", "brown", "gold",
    "silver", "glitter", "sparkle", "texture", "pattern", "vibrant", "calm", "peaceful",
    "energetic", "dynamic", "rustic", "elegant", "graceful", "charming", "quirky",
    "funny", "happy", "sad", "melancholic", "dramatic", "mysterious", "romantic",
    "adventure", "travel", "exploration", "monument", "castle", "fortress", "tower",
    "office", "interior", "room", "furniture", "decor", "design", "art", "creative",
    "craft", "handmade", "fashion", "style", "trend", "accessory", "jewelry",
    "clothing", "outfit", "model", "beauty", "makeup", "hair", "nails", "smile",
    "expression", "emotion", "family", "friends", "group", "celebration", "wedding",
    "birthday", "holiday", "vacation", "tourism", "scenic", "panorama", "drone", "aerial",
    "underwater", "macro", "wildlife", "bird", "cat", "horse", "elephant", "lion",
    "tiger", "bear", "monkey", "giraffe", "zebra", "snake", "insect", "butterfly",
    "flower", "plant", "bush", "leaf", "petal", "farm", "countryside", "field",
    "farmhouse", "barn", "harvest", "sunflower", "orchid", "rose", "daisy", "lily",
    "tulip", "floral", "patterned", "geometric", "symmetry", "balance", "contrast",
    "light", "shadow", "reflection", "mirror", "pool", "pond", "coast", "cliff",
    "rock", "hill", "valley", "cave", "volcano", "glacier", "frost", "mist", "fog",
    "haze", "storm", "lightning", "thunder", "breeze", "moon", "star", "galaxy",
    "universe", "space", "planet", "astronomy", "cosmos", "technology", "computer",
    "phone", "gadget", "robot", "machine", "engine", "circuit", "innovation",
    "future", "futuristic", "industrial", "building", "skyscraper", "factory",
    "warehouse", "laboratory", "science", "research", "education", "school",
    "university", "college", "library", "book", "paper", "writing", "artwork",
    "crafts", "sculpture", "exhibit", "instrument", "guitar", "piano", "drums",
    "violin", "saxophone", "band", "orchestra", "singing", "dance", "theater", "play",
    "comedy", "drama", "action", "horror", "romance", "thriller", "mystery", "crime",
    "history", "culture", "tradition", "heritage", "parade", "gathering", "safari",
    "camping", "hiking", "fishing", "roadtrip", "peak", "snowy", "icy", "retro",
    "modernist", "contemporary", "artisan", "handcrafted", "local", "organic", "exotic",
    "luxury", "budget", "detailed", "minimalist", "grand", "spacious", "cozy", "intimate",
    "bright", "dim", "warm", "cool", "vivid", "pastel", "rich", "subtle", "bold",
    "intricate", "delicate", "rough", "smooth", "polished", "rusty", "shiny", "glossy",
    "matte", "translucent", "transparent", "opaque", "reflective", "diffuse", "radiant",
    "luminous", "glowing", "sparkling", "shimmering", "effervescent", "timeless",
    "iconic", "legendary", "epic", "historic", "innovative", "experimental",
    "conceptual", "maximal", "eclectic", "diverse", "varied", "assorted", "mixed",
    "balanced", "harmonious", "discordant", "chaotic", "organized", "structured",
    "unstructured", "freeform", "representational", "surreal", "realistic",
    "photorealistic", "impressionistic", "expressionistic", "symbolic", "figurative",
    "nonfigurative", "urbanlife", "streetart", "graffiti", "skateboard", "bicycle",
    "motorcycle", "bus", "train", "airplane", "boat", "sailboat", "ship", "ferry",
    "cafe", "restaurant", "diner", "bar", "club", "carnival", "cooking", "recipe",
    "meal", "snack", "dessert", "chocolate", "fruit", "vegetable", "drink", "coffee",
    "tea", "beer", "wine", "cocktail", "barbecue", "grill", "picnic", "outdoors",
    "campfire", "sunbathe", "swim", "dive", "surf", "ski", "snowboard"
]

def remove_stopwords(text):
    """
    Removes stop words from the given text and returns a list of remaining words.

    Args:
        text (str): The input text.

    Returns:
        list: List of words without stop words.
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)  # Tokenize the text
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return filtered_words

def tag_image(image_path):
    """Generate tags for an image using the CLIP model."""
    image = Image.open(image_path).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize([f"a photo of {label}" for label in CANDIDATE_LABELS]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarities = (image_features @ text_features.T).squeeze(0)
    sim_scores = similarities.cpu().numpy().tolist()
    top_idxs = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)[:5]
    top_tags = [CANDIDATE_LABELS[i] for i in top_idxs]
    return list(set(top_tags))

def tag_video(video_path):
    """Extract key frames from video and generate tags using CLIP."""
    tags_set = set()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0, 0, 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_to_extract = 5
    step = max(1, frame_count // frames_to_extract)
    for frame_idx in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        _, buf = cv2.imencode(".jpg", frame)
        frame_image = Image.open(io.BytesIO(buf.tobytes())).convert("RGB")
        frame_tensor = clip_preprocess(frame_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(frame_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_inputs = clip.tokenize([f"a photo of {label}" for label in CANDIDATE_LABELS]).to(device)
            text_features = clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(0)
            sim_scores = similarities.cpu().numpy()
        top_idxs = sim_scores.argsort()[-3:][::-1]
        for idx in top_idxs:
            tags_set.add(CANDIDATE_LABELS[idx])
    cap.release()
    return list(tags_set), duration, width, height

def tag_audio(audio_path):
    """Detect music genre tags from an audio file using Musicnn."""
    tags = top_tags(audio_path, model='MTT_musicnn', topN=3)
    if isinstance(tags, tuple):
        tags = tags[0]
    tags = [t.lower() for t in tags]
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=60)
        duration = librosa.get_duration(y=y, sr=sr)
    except Exception:
        duration = 0
    return tags, duration

import os
from unstructured.partition.auto import partition  # Ensure `unstructured` is installed

import os
from unstructured.partition.auto import partition  # Ensure `unstructured` is installed

def load_document(file_path):
    """
    Loads and extracts text from a file (PDF, DOCX, TXT, PPT, etc.)
    using Unstructured's partition module.

    Args:
        file_path (str): Path to the document.

    Returns:
        str: Combined text extracted from all pages/documents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Extract text content from the document
        docs = partition(filename=file_path)
        extracted_text = " ".join(doc.text for doc in docs if hasattr(doc, "text") and doc.text)

        # If no text is extracted, return a placeholder message
        if not extracted_text.strip():
            print(f"⚠ No readable text found in {file_path}")
            return "No readable text found."

        print(f"✅ Extracted Text from {file_path}:\n{extracted_text[:500]}...\n")  # Log first 500 chars
        return extracted_text
    except Exception as e:
        print(f"❌ Error extracting text from {file_path}: {e}")
        return "Error processing document."


from unstructured.partition.auto import partition  # Import for text extraction

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads, tagging, and indexing in Elasticsearch."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    storage_filename = str(uuid.uuid4()) + "_" + file.filename
    blob = bucket.blob(storage_filename)
    blob.upload_from_file(file.stream, content_type=file.content_type)
    blob.make_public()
    file_url = blob.public_url

    content_type = file.content_type or ''
    media_type = None
    metadata = {}
    extracted_text = ""

    if content_type.startswith('image'):
        media_type = 'image'
    elif content_type.startswith('video'):
        media_type = 'video'
    elif content_type.startswith('audio'):
        media_type = 'audio'
    elif content_type.startswith('text') or file.filename.lower().endswith(('.pdf', '.docx', '.txt', '.ppt')):
        media_type = 'text'
    else:
        media_type = 'file'

    # Temporary file path for processing
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, storage_filename)
    blob.download_to_filename(temp_path)

    try:
        if media_type == 'image':
            tags = tag_image(temp_path)
            img = Image.open(temp_path)
            metadata['width'], metadata['height'] = img.size
            img.close()
        elif media_type == 'video':
            tags, duration, width, height = tag_video(temp_path)
            metadata.update({"width": width, "height": height, "duration": round(duration, 2)})
        elif media_type == 'audio':
            tags, duration = tag_audio(temp_path)
            metadata['duration'] = round(duration, 2)
        elif media_type == 'text':
            tags = []  # No specific tags, just extracting text
            extracted_text = load_document(temp_path)  # Extract text
            metadata["text_excerpt"] = extracted_text[:500]  # Store first 500 characters
            tags = remove_stopwords(extracted_text[:500])
        else:
            tags = []
    except Exception as e:
        print("Tagging error:", e)
        blob.delete()
        return jsonify({"error": f"Tagging failed: {str(e)}"}), 500

    # Store extracted text in metadata for Elasticsearch
    try:
        blob.metadata = {"tags": json.dumps(tags), "text_excerpt": extracted_text[:500]}
        blob.patch()
    except Exception as e:
        print(f"Failed to update Firebase metadata: {str(e)}")

    # Indexing in Elasticsearch
    doc = {
        "filename": file.filename,
        "storage_id": storage_filename,
        "url": file_url,
        "type": media_type,
        "tags": tags,
        "format": file.content_type or os.path.splitext(file.filename)[1],
        "metadata": metadata
    }
    es.index(index=INDEX_NAME, id=storage_filename, document=doc)
    es.indices.refresh(index=INDEX_NAME)

    return jsonify({"message": "File uploaded successfully", "data": doc}), 200





@app.route('/delete_all', methods=['POST'])
def delete_all():
    """
    Delete all files from Firebase Storage and clear the Elasticsearch index.
    Use with caution.
    """
    try:
        blobs = list(bucket.list_blobs())
        for blob in blobs:
            blob.delete()
        storage_msg = f"Deleted {len(blobs)} files from Firebase Storage."
    except Exception as e:
        storage_msg = f"Error deleting Firebase files: {str(e)}"
    
    try:
        if es.indices.exists(index=INDEX_NAME):
            es.indices.delete(index=INDEX_NAME)
        es.indices.create(index=INDEX_NAME, body=mapping)
        es_msg = "Elasticsearch index cleared and recreated."
    except Exception as e:
        es_msg = f"Error clearing Elasticsearch index: {str(e)}"
    
    print(storage_msg, es_msg)
    return jsonify({"message": f"{storage_msg} {es_msg}"}), 200

@app.route('/delete', methods=['POST'])
def delete_file():
    """
    Delete a file from Firebase Storage and remove its metadata from Elasticsearch.
    Provide either the 'storage_id' or the original 'filename' in the JSON payload.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided."}), 400

    storage_id = data.get("storage_id")
    filename = data.get("filename", "").strip()
    
    print(f"\n🔴 Deleting File - Storage ID: {storage_id}, Filename: {filename}\n")

    # If storage_id is missing, try finding it via filename
    if not storage_id and filename:
        query = {
            "query": {
                "match_phrase": {"filename.keyword": filename}  # Use keyword field for exact match
            }
        }
        try:
            res = es.search(index=INDEX_NAME, body=query, size=1)
            hits = res.get("hits", {}).get("hits", [])

            if hits:
                storage_id = hits[0]["_id"]
                print(f"✅ Found Storage ID: {storage_id} for Filename: {filename}")
            else:
                print(f"❌ No document found for filename '{filename}'")
                return jsonify({"error": f"No document found with filename '{filename}'."}), 404
        except Exception as e:
            print(f"❌ Elasticsearch query failed: {e}")
            return jsonify({"error": f"Elasticsearch query failed: {str(e)}"}), 500

    if not storage_id:
        print("❌ No valid storage_id provided")
        return jsonify({"error": "No storage_id provided."}), 400

    # Delete file from Firebase Storage
    try:
        blob = bucket.blob(storage_id)
        if blob.exists():
            blob.delete()
            print(f"✅ Deleted file from Firebase Storage: {storage_id}")
        else:
            print(f"⚠ File not found in Firebase Storage: {storage_id}")
    except Exception as e:
        print(f"❌ Failed to delete file from storage: {e}")
        return jsonify({"error": f"Failed to delete file from storage: {str(e)}"}), 500

    # Delete metadata from Elasticsearch
    try:
        es.delete(index=INDEX_NAME, id=storage_id)
        print(f"✅ Deleted metadata from Elasticsearch for: {storage_id}")
    except Exception as e:
        print(f"⚠ Failed to delete metadata from Elasticsearch: {e}")

    return jsonify({"message": f"File with storage_id '{storage_id}' deleted successfully."}), 200

# Initialize a Hugging Face text2text generation pipeline.
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=0 if torch.cuda.is_available() else -1)

def get_all_file_tags():
    """Retrieve all file IDs and their associated tags from Firebase Storage metadata."""
    files_with_tags = []
    try:
        blobs = list(bucket.list_blobs())
        for blob in blobs:
            metadata = blob.metadata or {}
            tags = json.loads(metadata.get("tags", "[]")) if isinstance(metadata.get("tags"), str) else []
            files_with_tags.append((blob.name, tags))
    except Exception as e:
        print(f"Error retrieving file metadata: {str(e)}")
    return files_with_tags

def llm_filter_results(query_text, candidate_tuples):
    """
    Uses a Hugging Face LLM to determine which candidate file IDs are most relevant
    to the given query based solely on the provided file names and tags.
    The prompt instructs the LLM to return the exact storage_ids as provided.
    
    Parameters:
        query_text (str): The user's search query.
        candidate_tuples (list of tuples): Each tuple is (file_id, filename, tags_string)
        
    Returns:
        list of file_ids (str) that are considered most relevant.
    """
    prompt = f"""You are an expert content matcher.
User query: "{query_text}".
Below is a list of files with their file storage_ids, file names, and associated tags.
Return only the file storage_ids (exactly as given) that best match the query based solely on these fields, as a comma-separated list.
Do not alter or abbreviate the storage_ids.

Files:
"""
    for file_id, filename, tags_str in candidate_tuples:
        prompt += f"- File ID: {file_id}. Filename: {filename}. Tags: {tags_str}\n"
    prompt += "\nAnswer (just the comma-separated exact file IDs):"
    
    output = llm_pipeline(prompt, max_length=200, do_sample=False)
    answer = output[0]['generated_text'].strip()
    filtered_ids = [fid.strip() for fid in answer.split(",") if fid.strip()]
    return filtered_ids

from nltk.corpus import wordnet
import nltk

# Download WordNet if not already installed
nltk.download("wordnet")

from nltk.corpus import wordnet
import nltk

# Download WordNet if not already installed
nltk.download("wordnet")

@app.route('/search', methods=['GET'])
def search_media():
    """Enhanced Search: Uses WordNet Synonyms, Fuzzy Matching, and Full-Text Search for Better Tag Matching."""
    query_text = request.args.get('q', '').strip().lower()
    media_type = request.args.get('type')
    min_res = request.args.get('min_resolution')
    min_dur = request.args.get('min_duration')

    print("\n🔍 Search Query:", query_text)

    filters = []
    if media_type:
        filters.append({"term": {"type": media_type}})
    if min_res:
        filters.append({"range": {"metadata.width": {"gte": int(min_res)}}})
    if min_dur:
        filters.append({"range": {"metadata.duration": {"gte": float(min_dur)}}})

    # Handle empty search query: Return all results
    if not query_text:
        es_query = {
            "bool": {
                "must": {"match_all": {}},
                "filter": filters
            }
        }
    else:
        # Expand search query using synonyms from WordNet
        expanded_queries = {query_text}
        for syn in wordnet.synsets(query_text):
            for lemma in syn.lemmas():
                expanded_queries.add(lemma.name().replace("_", " "))

        print("🔍 Expanded Search Terms:", expanded_queries)

        # Build Elasticsearch query with exact matches, fuzzy search, and extracted text lookup
        should_conditions = [
            {"match_phrase": {"tags": term}} for term in expanded_queries
        ] + [
            {"match": {"tags": {"query": query_text, "fuzziness": "AUTO"}}},  # Fuzzy match on tags
            {"match": {"metadata.text_excerpt": {"query": query_text, "operator": "and"}}}  # Full-text search in extracted text
        ]

        es_query = {
            "bool": {
                "should": should_conditions,
                "minimum_should_match": 1,
                "filter": filters
            }
        }

    try:
        results = es.search(index=INDEX_NAME, query=es_query, size=100)
        print(f"📌 Raw Elasticsearch Results (Total: {results['hits']['total']['value']}):", results)
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

    candidate_results = results.get('hits', {}).get('hits', [])

    # Ensure results are formatted properly
    final_results = []
    for hit in candidate_results:
        src = hit["_source"]
        file_id = src.get("storage_id", "")
        filename = src.get("filename", "Unnamed")
        tags = src.get("tags", [])

        # Ensure extracted text is included in results if available
        extracted_text = src.get("metadata", {}).get("text_excerpt", "")

        final_results.append({
            "storage_id": file_id,
            "filename": filename,
            "url": src.get("url", ""),
            "type": src.get("type", "unknown"),
            "tags": ", ".join(tags) if tags else "No tags available",
            "extracted_text": extracted_text[:500] + "..." if extracted_text else "No extracted text available"  # Limit preview to 500 chars
        })

    print("✅ Final Search Results:", final_results)

    return jsonify({"results": final_results}), 200





if __name__ == "__main__":
    app.run(debug=True)
