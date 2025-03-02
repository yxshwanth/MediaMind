import React, { useState, useEffect } from "react";
import { Container, AppBar, Toolbar, Typography, Button, Box } from "@mui/material";
import UploadForm from "./UploadForm";
import SearchBar from "./SearchBar";
import MediaGrid from "./MediaGrid";
import MediaModal from "./MediaModal";

function App() {
  const [mediaItems, setMediaItems] = useState([]);
  const [searchResults, setSearchResults] = useState(null);
  const [selectedMedia, setSelectedMedia] = useState(null);

  const fetchAllItems = async () => {
    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/search`);
      const data = await res.json();
      if (data.results) {
        setMediaItems(data.results);
      }
    } catch (err) {
      console.error("Failed to fetch media items", err);
    }
  };

  useEffect(() => {
    fetchAllItems();
  }, []);

  const handleDeleteAll = async () => {
    try {
      await fetch(`${process.env.REACT_APP_API_URL}/delete_all`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      fetchAllItems();
    } catch (error) {
      console.error("Failed to delete all items", error);
    }
  };

  const handleUploadSuccess = (newItem) => {
    setMediaItems((prev) => [newItem, ...prev]);
    setSearchResults(null);
  };

  const handleSearch = async (query, filters) => {
    let url = `${process.env.REACT_APP_API_URL}/search?q=${encodeURIComponent(query)}`;
    if (filters.type && filters.type !== "all") {
      url += `&type=${filters.type}`;
    }
    if (filters.minResolution) {
      url += `&min_resolution=${filters.minResolution}`;
    }
    if (filters.minDuration) {
      url += `&min_duration=${filters.minDuration}`;
    }
    try {
      const res = await fetch(url);
      const data = await res.json();
      if (data.results) {
        setSearchResults(data.results);
      }
    } catch (err) {
      console.error("Search failed", err);
    }
  };

  const handleSelectMedia = (item) => {
    setSelectedMedia(item);
  };

  const handleCloseModal = () => {
    setSelectedMedia(null);
  };

  const handleDeleteItem = (filename) => {
    setMediaItems((prev) => prev.filter(item => item.filename !== filename));
  };

  const itemsToDisplay = searchResults !== null ? searchResults : mediaItems;

  return (
    <div>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">Media Library</Typography>
        </Toolbar>
      </AppBar>
      <Container className="my-4">
        <Box className="d-flex flex-wrap align-items-center mb-3">
          <UploadForm onUploadSuccess={handleUploadSuccess} />
          <Button onClick={handleDeleteAll} variant="contained" color="error" className="ms-3">
            <i className="fas fa-trash-alt me-1"></i> Delete All
          </Button>
        </Box>
        <SearchBar onSearch={handleSearch} />
        <MediaGrid items={itemsToDisplay} onDelete={handleDeleteItem} onSelect={handleSelectMedia} />
      </Container>
      <MediaModal show={selectedMedia !== null} onHide={handleCloseModal} media={selectedMedia} />
    </div>
  );
}

export default App;
