import React, { useState } from "react";

function SearchBar({ onSearch }) {
  const [query, setQuery] = useState("");
  const [type, setType] = useState("all");
  const [minResolution, setMinResolution] = useState("");
  const [minDuration, setMinDuration] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    const filters = {
      type,
      minResolution: minResolution.trim(),
      minDuration: minDuration.trim(),
    };
    onSearch(query, filters);
  };

  // The autoComplete attribute is set to "off" to disable browser suggestions.
  return (
    <form onSubmit={handleSubmit} className="d-flex flex-wrap align-items-center gap-2 mb-3">
      <input
        type="text"
        className="form-control"
        placeholder="Search..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ maxWidth: "200px" }}
        autoComplete="off"
      />
      <select
        className="form-select"
        value={type}
        onChange={(e) => setType(e.target.value)}
        style={{ maxWidth: "150px" }}
      >
        <option value="all">All</option>
        <option value="image">Images</option>
        <option value="video">Videos</option>
        <option value="audio">Audio</option>
      </select>
      <input
        type="number"
        className="form-control"
        placeholder="Min Width (px)"
        value={minResolution}
        onChange={(e) => setMinResolution(e.target.value)}
        style={{ maxWidth: "150px" }}
      />
      <input
        type="number"
        className="form-control"
        placeholder="Min Duration (s)"
        value={minDuration}
        onChange={(e) => setMinDuration(e.target.value)}
        style={{ maxWidth: "150px" }}
      />
      <button type="submit" className="btn btn-primary">
        <i className="fas fa-search me-1"></i> Search
      </button>
      <button type="button" className="btn btn-secondary" onClick={() => {
        setQuery("");
        setType("all");
        setMinResolution("");
        setMinDuration("");
        onSearch("", { type: "all", minResolution: "", minDuration: "" });
      }}>
        Clear
      </button>
    </form>
  );
}

export default SearchBar;
