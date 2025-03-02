import React from 'react';
import { Modal, Button } from 'react-bootstrap';

function MediaModal({ show, onHide, media }) {
  if (!media) return null;
  const { type, url, filename, tags, metadata } = media;
  
  return (
    <Modal show={show} onHide={onHide} size="lg" centered>
      <Modal.Header closeButton>
        <Modal.Title>{filename}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {type === 'image' && (
          <img src={url} alt={filename} className="img-fluid" style={{ width: '100%' }} />
        )}
        {type === 'video' && (
          <video src={url} controls className="img-fluid" style={{ width: '100%' }} />
        )}
        {type === 'audio' && (
          <audio src={url} controls className="w-100" />
        )}
        <div className="mt-3">
          <p><strong>Type:</strong> {type}</p>
          {metadata.width && <p><strong>Resolution:</strong> {metadata.width}x{metadata.height}px</p>}
          {metadata.duration && <p><strong>Duration:</strong> {metadata.duration}s</p>}
          <p><strong>Tags:</strong> {tags && tags.join(', ')}</p>
        </div>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>Close</Button>
      </Modal.Footer>
    </Modal>
  );
}

export default MediaModal;
