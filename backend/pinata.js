// pinata.js
const express = require('express');
const pinataSDK = require('pinata-sdk');
require('dotenv').config();

const pinata = pinataSDK(process.env.PINATA_API_KEY, process.env.PINATA_API_SECRET);
const router = express.Router(); // Use Router for modular routing

// POST endpoint to upload Python code to Pinata
router.post('/upload-python-code', async (req, res) => {
    const { code } = req.body; // Extract Python code from the request

    try {
        const buffer = Buffer.from(code, 'utf-8'); // Convert code to buffer
        const result = await pinata.pinFileToIPFS(buffer); // Upload code to Pinata
        const ipfsHash = result.IpfsHash; // Get IPFS hash

        res.status(201).json({ message: 'Python code uploaded successfully', ipfsHash });
    } catch (error) {
        console.error("Error uploading to Pinata:", error);
        res.status(500).json({ error: error.message });
    }
});

// Export the router to be used in server.js
module.exports = router;
