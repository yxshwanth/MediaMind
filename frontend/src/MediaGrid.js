import React from 'react';

function MediaGrid({ items, onDelete, onSelect }) {
  if (!items || items.length === 0) {
    return <p className="text-center mt-4">No media found.</p>;
  }
  return (
    <div className="container">
      {/* Using Bootstrap 5 grid to force 3 columns on medium and larger screens */}
      <div className="row row-cols-1 row-cols-md-3 g-3">
        {items.map((item, index) => (
          <div key={index} className="col">
            <MediaItem item={item} onDelete={onDelete} onSelect={onSelect} />
          </div>
        ))}
      </div>
    </div>
  );
}

function MediaItem({ item, onDelete, onSelect }) {
  const { type, url, filename, tags, metadata } = item;
  let mediaPreview;
  if (type === 'image') {
    mediaPreview = (
      <img
        src={url}
        alt={filename}
        className="img-fluid"
        style={{ width: '100%', height: '150px', objectFit: 'cover' }}
      />
    );
  } else if (type === 'video') {
    mediaPreview = (
      <video
        src={url}
        controls
        className="img-fluid"
        style={{ width: '100%', height: '150px', objectFit: 'cover' }}
      />
    );
  } else if (type === 'audio') {
    // Display a clickable headphone icon for audio files
    mediaPreview = (
      <div
        style={{
          width: '100%',
          height: '150px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#e9ecef',
          cursor: 'pointer'
        }}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(item);
        }}
      >
        <i className="fas fa-headphones" style={{ fontSize: '3rem', color: '#6c757d' }}></i>
      </div>
    );
  } else {
    mediaPreview = (
      <a href={url} target="_blank" rel="noopener noreferrer">
        {filename}
      </a>
    );
  }
  const tagList = tags && tags.length ? tags.join(', ') : 'No tags';

  const handleDelete = async (e) => {
    e.stopPropagation(); // Prevent triggering selection when deleting
    if (window.confirm(`Are you sure you want to delete ${filename}?`)) {
      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL}/delete`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename }),
        });
        const data = await response.json();
        if (response.ok) {
          alert(data.message);
          if (onDelete) {
            onDelete(filename);
          }
        } else {
          alert(data.error || "Failed to delete file.");
        }
      } catch (error) {
        console.error("Delete error:", error);
        alert("Error deleting file.");
      }
    }
  };

  const handleSelect = () => {
    if (onSelect) {
      onSelect(item);
    }
  };

  return (
    <div className="card h-100" onClick={handleSelect} style={{ cursor: 'pointer' }}>
      <div className="card-img-top">{mediaPreview}</div>
      <div className="card-body">
        <h6 className="card-title text-truncate" title={filename}>
          {filename}
        </h6>
        <p className="card-text small mb-2">
          <strong>Type:</strong> {type} <br />
          {metadata.width && (
            <>
              <strong>Resolution:</strong> {metadata.width}x{metadata.height}px <br />
            </>
          )}
          {metadata.duration && (
            <>
              <strong>Duration:</strong> {metadata.duration}s <br />
            </>
          )}
          <strong>Tags:</strong> {tagList}
        </p>
        <button onClick={handleDelete} className="btn btn-danger btn-sm">
          <i className="fas fa-trash-alt"></i> Delete
        </button>
      </div>
    </div>
  );
}

export default MediaGrid;
