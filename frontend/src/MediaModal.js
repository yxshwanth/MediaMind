import React from "react";
import { Modal, Box, Typography } from "@mui/material";

function MediaModal({ show, onHide, media }) {
  if (!media) return null;

  let mediaContent;
  if (media.type === "image") {
    mediaContent = <img src={media.url} alt={media.filename} style={{ maxWidth: "100%", maxHeight: "400px" }} />;
  } else if (media.type === "video") {
    mediaContent = <video src={media.url} controls style={{ maxWidth: "100%", maxHeight: "400px" }} />;
  } else if (media.type === "audio") {
    mediaContent = <audio src={media.url} controls />;
  }else if (media.type === "text") {
      mediaContent = <audio src={media.url} alt={media.filename} style={{ maxWidth: "100%", maxHeight: "400px" }} />;
  } else {
    mediaContent = <p>Unsupported file type.</p>;
  }

  return (
    <Modal open={show} onClose={onHide}>
      <Box className="modal-box">
        <Typography variant="h6">{media.filename}</Typography>
        {mediaContent}
      </Box>
    </Modal>
  );
}

export default MediaModal;
