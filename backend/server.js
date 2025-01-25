const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

// Import the cron job (which will start running automatically)
require('./cronjob'); // Ensure this is at the top to run the cron job

const app = express();
app.use(cors());
app.use(express.json());


// Endpoint to check Python syntax
app.post('/check-python-syntax', (req, res) => {
    const pythonCode = req.body.code;

    if (!pythonCode) {
        return res.status(400).json({ error: 'No Python code provided' });
    }

    const tempFile = path.join(__dirname, 'temp_script.py');
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
                code: pythonCode, // Return the Python code
            });
        }
    });
});

// Start the server
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
