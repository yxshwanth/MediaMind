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

  /** Fetch all media items from backend */
  const fetchAllItems = async () => {
    try {
        const res = await fetch(`${process.env.REACT_APP_API_URL}/search`);
        const data = await res.json();

        console.log("Fetched media items:", data);

        if (data.results && Array.isArray(data.results) && data.results.length > 0) {
            setMediaItems(data.results);
            setSearchResults(null); // Reset search results when fetching all items
        } else {
            console.warn("No media found in the backend.");
            setMediaItems([]); // Clear UI if no media is found
        }
    } catch (err) {
        console.error("Failed to fetch media items", err);
    }
};


  // Fetch all media items on initial load
  useEffect(() => {
    fetchAllItems();
  }, []);

  /** Deletes all media from backend & updates UI */
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

  /** Adds newly uploaded file to media list */
  const handleUploadSuccess = (newItem) => {
    console.log("New file uploaded:", newItem);
    setMediaItems((prev) => [newItem, ...prev]);
    setSearchResults(null);
  };

  /** Handles search functionality */
  const handleSearch = async (query, filters) => {
    if (!query.trim()) {
      fetchAllItems(); // Reset to all items if search is empty
      return;
    }

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
      console.log("Search results:", data);

      if (data.results && data.results.length > 0) {
        // Filter the results to only show exact matches
        const filteredResults = data.results.filter((item) => {
          return (
            item.tags &&
            item.tags.toLowerCase().split(", ").includes(query.toLowerCase())
          );
        });

        if (filteredResults.length > 0) {
          setSearchResults(filteredResults);
        } else {
          setSearchResults([]); // No matching media
        }
      } else {
        setSearchResults([]); // No matching media
      }
    } catch (err) {
      console.error("Search failed", err);
      setSearchResults([]); // Ensure UI updates even if there's an error
    }
  };

  /** Handles media selection (opens in modal) */
  const handleSelectMedia = (item) => {
    setSelectedMedia(item);
  };

  /** Closes the modal */
  const handleCloseModal = () => {
    setSelectedMedia(null);
  };

  /** Deletes an individual media item */
  const handleDeleteItem = async (filename) => {
    try {
      await fetch(`${process.env.REACT_APP_API_URL}/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename }),
      });
      setMediaItems((prev) => prev.filter(item => item.filename !== filename));
    } catch (error) {
      console.error("Failed to delete item", error);
    }
  };

  /** Determine what to display */
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
        {/* If no items found, display message */}
        {itemsToDisplay.length === 0 ? (
          <p className="text-center mt-4">No media found.</p>
        ) : (
          <MediaGrid items={itemsToDisplay} onDelete={handleDeleteItem} onSelect={handleSelectMedia} />
        )}
      </Container>
      <MediaModal show={selectedMedia !== null} onHide={handleCloseModal} media={selectedMedia} />
    </div>
  );
}

export default App;
