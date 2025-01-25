const express = require('express');
const axios = require('axios');
const FormData = require('form-data');
require('dotenv').config({ path: '../.env' });

const router = express.Router();

router.post('/upload-python-code', async (req, res) => {
  const { code } = req.body; // The Python code from the client
  console.log(code);
  console.log("Upload API called");

  if (!code) {
    return res.status(400).json({ error: 'No code provided' });
  }

  try {
    // Create a new FormData instance
    const data = new FormData();

    // Convert Python code into a Buffer and add it as a file to the FormData
    const buffer = Buffer.from(code, 'utf-8'); // Use Buffer instead of Blob for Node.js
    const file = new require('stream').Readable();
    file._read = () => {}; // Override _read method, which is necessary for streams
    file.push(buffer);
    file.push(null); // Push null to indicate the end of the stream

    // Append the file to FormData
    data.append("file", file, "python-code.py");

    // Send the file to Pinata
    const response = await axios({
      method: 'post',
      url: 'https://api.pinata.cloud/pinning/pinFileToIPFS', // Use the correct URL for Pinata's API
      headers: {
        ...data.getHeaders(), // Important: Pass the correct headers for FormData
        'Authorization': `Bearer ${process.env.PINATA_JWT}`, // Using JWT for authentication
      },
      data: data,
    });

    // Respond with the IPFS hash
    console.log("Uploaded succesfully");
    res.status(201).json({ 
      message: 'Code uploaded successfully', 
      ipfsHash: response.data.IpfsHash 
    });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.response?.data || error.message });
  }
});

module.exports = router;
