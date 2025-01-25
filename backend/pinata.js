// pinata.js
const express = require('express');
const multer = require('multer');
const { PinataSDK } = require('pinata-web3');
require('dotenv').config();

// Initialize the Pinata SDK
const pinata = new PinataSDK({
  pinataJwt: process.env.PINATA_JWT,
  pinataGateway: process.env.GATEWAY_URL,
});

const router = express.Router(); // Use Router for modular routing

// Setup multer for file handling
const storage = multer.memoryStorage();  // Store the file in memory
const upload = multer({ storage: storage });  // Create multer instance with memory storage

// POST endpoint to upload Python code to Pinata
router.post('/upload-python-code', upload.single('file'), async (req, res) => {
  const file = req.file;

  if (!file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  try {
    // Create a buffer from the uploaded file
    const buffer = file.buffer;

    // Upload the file to Pinata
    const upload = await pinata.upload.file(buffer);

    // Get the IPFS hash from the upload result
    const ipfsHash = upload.IpfsHash;

    // Respond with the IPFS hash
    res.status(201).json({ message: 'Python code uploaded successfully', ipfsHash });
  } catch (error) {
    console.error('Error uploading to Pinata:', error);
    res.status(500).json({ error: error.message });
  }
});

// Export the router to be used in server.js
module.exports = router;
