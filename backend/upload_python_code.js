const express = require('express');
const axios = require('axios');
const FormData = require('form-data');
const mysql = require('mysql2/promise');
require('dotenv').config({ path: '../.env' });

const router = express.Router();

// Database connection setup
const db = mysql.createPool({
  host: '127.0.0.1',
  user: 'root',
  password: 'Prateush@04',
  database: 'gpudb',
  port: 3306,
});

// New endpoint for uploading to DB first, then IPFS
router.post('/upload-to-db-then-ipfs', async (req, res) => {
  const { code, userId, details } = req.body; // The Python code, userId, and details from the client
  console.log("Upload to IPFS then DB called");

  if (!code) {
    return res.status(400).json({ error: 'No code provided' });
  }

  try {
    // Step 1: Insert the Python code into the database
    const query = 'INSERT INTO python_code (user_id, code_hash, details) VALUES (?, ?, ?)';
    const [result] = await db.query(query, [userId, '', details || 'test string']); // Insert without code_hash initially

    const recordId = result.insertId;
    console.log('Record inserted into python_code table, ID: ', recordId);

    // Step 2: Upload the Python code to IPFS (Pinata)
    const data = new FormData();
    const buffer = Buffer.from(code, 'utf-8');
    const file = new require('stream').Readable();
    file._read = () => {}; 
    file.push(buffer);
    file.push(null);

    data.append("file", file, "python-code.py");

    const pinataResponse = await axios({
      method: 'post',
      url: 'https://api.pinata.cloud/pinning/pinFileToIPFS',
      headers: {
        ...data.getHeaders(),
        'Authorization': `Bearer ${process.env.PINATA_JWT}`,
      },
      data: data,
    });

    const ipfsHash = pinataResponse.data.IpfsHash;
    console.log("Uploaded to IPFS, IPFS Hash: ", ipfsHash);

    // Step 3: Update the database record with the IPFS hash
    await db.query('UPDATE python_code SET code_hash = ? WHERE id = ?', [ipfsHash, recordId]);
    console.log('Database updated with IPFS hash');

    // Step 4: Respond with success
    res.status(201).json({
      message: 'Code uploaded to DB and IPFS successfully',
      ipfsHash: ipfsHash,
      recordId: recordId
    });

  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.response?.data || error.message });
  }
});

module.exports = router;
