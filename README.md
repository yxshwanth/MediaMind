# MediaMind

## Overview
The **AI-Powered Media Library** is a full-stack web application that enables users to upload, manage, and search multimedia files (images, videos, audio, documents) with automatic tagging, text extraction, and intelligent search capabilities powered by **OpenAI's CLIP**, **Musicnn**, **Hugging Face**, and **Elasticsearch**.

## Features
✅ **Automatic Tagging**: AI-generated tags for images, videos, and audio using **OpenAI's CLIP** and **Musicnn**.  
✅ **Intelligent Search**: Full-text search with **WordNet Synonyms**, **Fuzzy Matching**, and **Elasticsearch**.  
✅ **Extracted Text Search**: OCR & text extraction from PDFs, Word, Excel, and PPT files.  
✅ **Firebase Storage**: Securely store and retrieve media files with public URLs.  
✅ **Multi-format Support**: Handles **images (PNG, JPG, WEBP)**, **videos (MP4, AVI)**, **audio (MP3, WAV)**, and **documents (PDF, DOCX, TXT, CSV, PPT)**.  
✅ **Responsive UI**: Built with **React, Bootstrap, and Flask API**.

## Tech Stack
- **Backend**: Python, Flask, Firebase Admin SDK, Elasticsearch, Hugging Face, OpenAI CLIP, Musicnn, Unstructured.io
- **Frontend**: React.js, Bootstrap
- **Database**: Elasticsearch
- **Storage**: Firebase Storage

## Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Node.js 14+
- Firebase account with Storage enabled
- Elasticsearch 7.10+

### Backend Setup (Flask API)
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/media-library.git
   cd media-library/backend
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Configure Firebase & Environment Variables:
   - Create a `firebase_credentials.json` file with Firebase credentials.
   - Create a `.env` file:
     ```sh
     FIREBASE_CRED_FILE=firebase_credentials.json
     FIREBASE_BUCKET=your-bucket-name
     ELASTIC_HOST=localhost:9200
     ELASTIC_USER=your-user
     ELASTIC_PASS=your-pass
     ```
5. Start the Flask server:
   ```sh
   python app.py
   ```

### Frontend Setup (React App)
1. Navigate to the frontend directory:
   ```sh
   cd ../frontend
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the development server:
   ```sh
   npm start
   ```

## Usage
- **Upload Files**: Click the "Browse" button to upload images, videos, audio, or documents.
- **Search**: Use keywords to search through media files with AI-assisted tagging.
- **Delete Media**: Click the delete button to remove any uploaded file.
- **View Extracted Text**: Click on text files (PDFs, DOCX, etc.) to see extracted content.

## API Endpoints
### Upload File
```http
POST /upload
```
Uploads a file and returns metadata, including AI-generated tags.

### Search Media
```http
GET /search?q=<query>
```
Performs an intelligent search using AI tagging, synonyms, and fuzzy matching.

### Delete File
```http
POST /delete
```
Removes a file from Firebase and Elasticsearch.

### Delete All Files
```http
POST /delete_all
```
⚠ **Warning**: Clears all stored files and metadata.

## Roadmap
🚀 Future improvements include:
- 🔍 **Enhanced OCR** for handwritten text.
- 🎞️ **Thumbnail generation** for videos.
- 🎼 **Audio transcription** for speech-to-text.

## Contributing
1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request 🚀

## License
This project is licensed under the MIT License.


