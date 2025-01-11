const cron = require('node-cron');
const axios = require('axios'); // To make API requests
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

// Function to execute Python code
async function executePythonCode() {
    try {
        // Fetch the latest Python code from the server's /check-python-syntax endpoint
        const response = await axios.post('http://localhost:5000/check-python-syntax', {
            code: 'print("This is the Python code executed by the cron job!")' // Send Python code dynamically
        });

        // Log the response to check if it's correct
        console.log('API Response:', response.data);

        // Check if the response is valid and contains the expected code
        if (response.data.valid && response.data.code) {
            const pythonCode = response.data.code; // Get Python code from the response

            // Log the code to verify
            console.log('Python Code:', pythonCode);

            // Write the code to a temporary file
            const tempFile = path.join(__dirname, 'temp_script.py');
            fs.writeFileSync(tempFile, pythonCode);

            // Execute the Python code
            exec(`python ${tempFile}`, (error, stdout, stderr) => {
                fs.unlinkSync(tempFile); // Clean up the temporary file

                if (error) {
                    console.error('Error executing Python code:', stderr.trim());
                } else {
                    console.log('Python code executed successfully');
                    console.log('Output:', stdout.trim());
                }
            });
        } else {
            console.log('Python code is invalid or missing.');
        }
    } catch (error) {
        console.error('Error fetching Python code:', error.message);
    }
}

// Schedule the task to run every minute
cron.schedule('* * * * *', () => {
    console.log('Running scheduled Python code execution...');
    executePythonCode();
});

console.log('Cron job is running...');
