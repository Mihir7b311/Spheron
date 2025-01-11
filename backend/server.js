const express = require('express');
const cors = require('cors'); // Import CORS

const app = express();

// Enable CORS for all origins
app.use(cors());
app.use(express.json()); // Middleware to parse JSON bodies

const { exec } = require('child_process');
const fs = require('fs');

// Endpoint to check Python syntax
app.post('/check-python-syntax', (req, res) => {
    const pythonCode = req.body.code;

    if (!pythonCode) {
        return res.status(400).json({ error: 'No Python code provided' });
    }

    const tempFile = 'temp_script.py';
    fs.writeFileSync(tempFile, pythonCode);

    exec(`python -m py_compile ${tempFile}`, (error, stdout, stderr) => {
        fs.unlinkSync(tempFile); // Clean up the temporary file

        if (error) {
            return res.status(400).json({
                valid: false,
                error: stderr.trim(), // Return the syntax error message
            });
        } else {
            return res.status(200).json({
                valid: true,
                message: 'Python code is valid',
            });
        }
    });
});

// Start the server
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
