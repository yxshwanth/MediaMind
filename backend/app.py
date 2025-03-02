import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib
import io
# Firebase Admin SDK for Storage
import firebase_admin
from firebase_admin import credentials, storage

# ElasticSearch client
from elasticsearch import Elasticsearch

# For image and video processing
import torch
import clip  # OpenAI's CLIP package
from PIL import Image
import numpy as np
import cv2

# For audio tagging using Musicnn
from musicnn.tagger import top_tags
import librosa

# Load environment variables from .env file (use python-dotenv if desired)
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for development; restrict in production

# Firebase configuration
FIREBASE_CRED_FILE = os.environ.get('FIREBASE_CRED_FILE', 'firebase_credentials.json')
FIREBASE_BUCKET    = os.environ.get('FIREBASE_BUCKET', '<your-bucket-name>')
cred = credentials.Certificate(FIREBASE_CRED_FILE)
firebase_admin.initialize_app(cred, {
    'storageBucket': FIREBASE_BUCKET
})
bucket = storage.bucket()

# ElasticSearch configuration
ELASTIC_HOST = os.environ.get('ELASTIC_HOST', 'localhost:9200')
ELASTIC_USER = os.environ.get('ELASTIC_USER')
ELASTIC_PASS = os.environ.get('ELASTIC_PASS')
if ELASTIC_USER and ELASTIC_PASS:
    es = Elasticsearch([ELASTIC_HOST], basic_auth=(ELASTIC_USER, ELASTIC_PASS))
else:
    es = Elasticsearch([ELASTIC_HOST])
INDEX_NAME = "media_files"

# Load CLIP model (ViT-B/32)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Candidate labels for zero-shot image tagging (adjust or expand as needed)
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


def tag_image(image_path):
    """Generate tags for an image using the CLIP model."""
    image = Image.open(image_path).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize([f"a photo of {label}" for label in CANDIDATE_LABELS]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)
    # Normalize features
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

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint to upload a media file, tag it, and index metadata."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = str(uuid.uuid4()) + "_" + file.filename
    blob = bucket.blob(filename)
    blob.upload_from_file(file.stream, content_type=file.content_type)
    blob.make_public()
    file_url = blob.public_url

    content_type = file.content_type or ''
    media_type = None
    tags = []
    metadata = {}

    try:
        if content_type.startswith('image'):
            media_type = 'image'
            temp_path = f"/tmp/{filename}"
            blob.download_to_filename(temp_path)
            tags = tag_image(temp_path)
            img = Image.open(temp_path)
            metadata['width'], metadata['height'] = img.size
            img.close()
        elif content_type.startswith('video'):
            media_type = 'video'
            temp_path = f"/tmp/{filename}"
            blob.download_to_filename(temp_path)
            tags, duration, width, height = tag_video(temp_path)
            metadata['width'] = width
            metadata['height'] = height
            metadata['duration'] = round(duration, 2)
        elif content_type.startswith('audio') or content_type in ['application/octet-stream', 'audio/mpeg']:
            media_type = 'audio'
            temp_path = f"/tmp/{filename}"
            blob.download_to_filename(temp_path)
            tags, duration = tag_audio(temp_path)
            metadata['duration'] = round(duration, 2)
        else:
            media_type = 'file'
            tags = []
    except Exception as e:
        blob.delete()
        return jsonify({"error": f"Tagging failed: {str(e)}"}), 500

    doc = {
        "filename": file.filename,
        "url": file_url,
        "type": media_type,
        "tags": tags,
        "format": file.content_type or os.path.splitext(file.filename)[1],
        "metadata": metadata
    }
    es.index(index=INDEX_NAME, document=doc)
    return jsonify({"message": "File uploaded successfully", "data": doc}), 200

@app.route('/delete_all', methods=['POST'])
def delete_all():
    """
    Delete all files from Firebase Storage and clear the Elasticsearch index.
    Use with caution.
    """
    # Delete all blobs in Firebase Storage
    try:
        blobs = list(bucket.list_blobs())
        for blob in blobs:
            blob.delete()
        storage_msg = f"Deleted {len(blobs)} files from Firebase Storage."
    except Exception as e:
        storage_msg = f"Error deleting Firebase files: {str(e)}"
    
    # Delete all documents from the Elasticsearch index by deleting the index and recreating it.
    try:
        if es.indices.exists(index=INDEX_NAME):
            es.indices.delete(index=INDEX_NAME)
        # Optionally, recreate the index with your mapping:
        es.indices.create(index=INDEX_NAME)
        es_msg = "Elasticsearch index cleared."
    except Exception as e:
        es_msg = f"Error clearing Elasticsearch index: {str(e)}"
    
    return jsonify({"message": f"{storage_msg} {es_msg}"}), 200


@app.route('/delete', methods=['POST'])
def delete_file():
    """
    Delete a file from Firebase Storage and remove its metadata from Elasticsearch.
    Expects a JSON payload with the 'filename' key.
    """
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided."}), 400

    filename = data['filename']
    
    # Delete the file from Firebase Storage
    try:
        blob = bucket.blob(filename)
        blob.delete()
    except Exception as e:
        return jsonify({"error": f"Failed to delete file from storage: {str(e)}"}), 500

    # Optionally, remove the file's metadata from Elasticsearch
    try:
        es.delete(index=INDEX_NAME, id=filename)
    except Exception as e:
        # Log the error, but you might choose to continue if the file deletion was successful
        print(f"Failed to delete metadata from Elasticsearch: {str(e)}")

    return jsonify({"message": f"File '{filename}' deleted successfully."}), 200


@app.route('/search', methods=['GET'])
def search_media():
    """Endpoint to search for media files by text query and filters."""
    query_text = request.args.get('q', '')
    media_type = request.args.get('type')
    min_res = request.args.get('min_resolution')
    min_dur = request.args.get('min_duration')

    must_clauses = []
    filter_clauses = []

    if query_text:
        must_clauses.append({
            "multi_match": {
                "query": query_text,
                "fields": ["tags^2", "filename"]
            }
        })
    else:
        must_clauses.append({"match_all": {}})

    if media_type:
        filter_clauses.append({"term": {"type": media_type}})
    if min_res:
        filter_clauses.append({"range": {"metadata.width": {"gte": int(min_res)}}})
    if min_dur:
        filter_clauses.append({"range": {"metadata.duration": {"gte": float(min_dur)}}})

    es_query = {
        "bool": {
            "must": must_clauses,
            "filter": filter_clauses
        }
    }
    try:
        results = es.search(index=INDEX_NAME, query=es_query, size=50)
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

    hits = results.get('hits', {}).get('hits', [])
    output = [hit["_source"] for hit in hits]
    return jsonify({"results": output}), 200

if __name__ == "__main__":
    app.run(debug=True)
