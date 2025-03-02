import React, { useState } from "react";

function MediaGrid({ items, onDelete, onSelect }) {
  if (!items || items.length === 0) {
    return <p className="text-center mt-4">No media found.</p>;
  }

  return (
    <div className="container">
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
  const { type, url, filename, tags, metadata, storage_id } = item;
  const [showTags, setShowTags] = useState(false); // Toggle for text file tags

  // Debugging: Check if type, URL, and storage_id exist correctly
  console.log("Rendering item:", item);

  let mediaPreview;

  if (type?.toLowerCase() === "image") {
    mediaPreview = (
      <img
        src={url}
        alt={filename}
        className="img-fluid"
        style={{ width: "100%", height: "180px", borderRadius: "8px" }}
      />
    );
  } else if (type?.toLowerCase() === "video") {
    mediaPreview = (
      <video
        src={url}
        controls
        className="img-fluid"
        style={{ width: "100%", height: "180px", borderRadius: "8px" }}
      />
    );
  } else if (type?.toLowerCase() === "audio") {
    mediaPreview = <audio src={url} controls style={{ width: "100%" }} />;
  } else if (type?.toLowerCase() === "text" || type?.toLowerCase() === "file") {
    mediaPreview = (
      <a href={url} target="_blank" rel="noopener noreferrer">
        {filename}
      </a>
    );
  } else {
    mediaPreview = (
      <div className="text-muted">
        <i className="fas fa-file-alt"></i> {filename}
      </div>
    );
  }

  // Process metadata safely
  const tagList = Array.isArray(tags) ? tags.join(", ") : tags || "No tags";
  const resolution =
    metadata?.width && metadata?.height
      ? `${metadata.width}x${metadata.height}px`
      : "N/A";
  const duration = metadata?.duration ? `${metadata.duration}s` : "N/A";
  const textExcerpt = metadata?.text_excerpt || "No extracted text available.";

  // Handle delete action
  const handleDelete = async (e) => {
    e.stopPropagation();
    console.log("🗑 Attempting to delete:", { filename, storage_id });

    if (!storage_id) {
      alert("Error: Missing storage ID for deletion.");
      return;
    }

    if (window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL}/delete`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ storage_id, filename }),
        });

        const data = await response.json();
        console.log("🗑 Delete API Response:", data);

        if (response.ok) {
          alert(data.message);
          if (onDelete) {
            onDelete(filename);
          }
        } else {
          alert(data.error || "Failed to delete file.");
        }
      } catch (error) {
        console.error("❌ Delete error:", error);
        alert("Error deleting file.");
      }
    }
  };

  return (
    <div
      className="card h-100 shadow-sm"
      style={{ borderRadius: "10px", overflow: "hidden", cursor: "pointer" }}
      onClick={() => {
        if (type?.toLowerCase() === "text" || type?.toLowerCase() === "file") {
          console.log(`📄 Extracted Text for ${filename}:`, textExcerpt);
          setShowTags(!showTags);
        } else {
          onSelect(item);
        }
      }}
    >
      <div className="card-img-top">{mediaPreview}</div>
      <div className="card-body">
        <h6 className="card-title text-truncate" title={filename}>
          {filename}
        </h6>
        <p className="card-text small mb-2">
          <strong>Type:</strong> {type} <br />
          {metadata?.width && metadata?.height && (
            <>
              <strong>Resolution:</strong> {resolution} <br />
            </>
          )}
          {metadata?.duration && (
            <>
              <strong>Duration:</strong> {duration} <br />
            </>
          )}

          {type?.toLowerCase() === "text" || type?.toLowerCase() === "file" ? (
            <>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowTags(!showTags);
                }}
                className="btn btn-info btn-sm"
              >
                {showTags ? "Hide Text" : "View Extracted Text"}
              </button>
              {showTags && (
                <p className="mt-2">
                  <strong>Extracted Text:</strong> {textExcerpt}
                </p>
              )}
            </>
          ) : (
            <p>
              <strong>Tags:</strong> {tagList}
            </p>
          )}
        </p>

        <button onClick={handleDelete} className="btn btn-danger btn-sm">
          <i className="fas fa-trash-alt"></i> Delete
        </button>
      </div>
    </div>
  );
}

export default MediaGrid;
