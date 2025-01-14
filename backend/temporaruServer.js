const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

// Import the cron job (which will start running automatically)
//require('./cronjob'); // Ensure this is at the top to run the cron job
require('./scheduleConfiguration.js');
const { scheduleDailyTask, scheduleWeeklyTask, scheduleMonthlyTask } = require('./scheduleConfiguration');

const app = express();
app.use(cors());
app.use(express.json());
const cron = require('node-cron');
const customData = {
    data: "Hmm cron job"
};
// cron.schedule('* * * * *', () => {
//     console.log("Cron job is running every minute to keep things active.");
// });
// Define the folder for temporary files
const TEMP_FOLDER = path.join(__dirname, 'temp');
const GPU_PYTHON_FOLDER = path.join(__dirname, 'GPUPython');

// Ensure the folders exist
if (!fs.existsSync(TEMP_FOLDER)) {
    fs.mkdirSync(TEMP_FOLDER, { recursive: true });
}
if (!fs.existsSync(GPU_PYTHON_FOLDER)) {
    fs.mkdirSync(GPU_PYTHON_FOLDER, { recursive: true });
}

// Track file count for GPUPython folder
let fileCounter = 0;

// Endpoint to check Python syntax
app.post('/check-python-syntax', (req, res) => {
    const pythonCode = req.body.code;
    const schedule = req.body.schedule;

    if (!pythonCode) {
        return res.status(400).json({ error: 'No Python code provided' });
    }

    const tempFile = path.join(TEMP_FOLDER, 'temp_script.py');
    fs.writeFileSync(tempFile, pythonCode);

    // Create a new file in the GPUPython folder
    fileCounter += 1;
    const gpuPythonFile = path.join(GPU_PYTHON_FOLDER, `code${fileCounter}.py`);
    fs.writeFileSync(gpuPythonFile, pythonCode);

    const gpuPythonScheduleFile = path.join(GPU_PYTHON_FOLDER, `code${fileCounter}.json`);
    fs.writeFileSync(gpuPythonScheduleFile, JSON.stringify(schedule, null, 2));

    exec(`python -m py_compile ${tempFile}`, (error, stdout, stderr) => {
        fs.unlinkSync(tempFile); // Clean up the temporary file

        if (error) {
            return res.status(400).json({
                valid: false,
                error: stderr.trim(), // Return the syntax error message
            });
        } else {
            const scheduleType = schedule.type;

            console.log(schedule.type);
            if (scheduleType === 'daily') {
                scheduleDailyTask(schedule.configuration,pythonCode,JSON.stringify(schedule, null, 2));
            } else if (scheduleType === 'monthly') {
                scheduleMonthlyTask(schedule.configuration,pythonCode,JSON.stringify(schedule, null, 2));
            } else if (scheduleType === 'weekly') {
                scheduleWeeklyTask(schedule.configuration,pythonCode,JSON.stringify(schedule, null, 2));
            }

            console.log("Schedule is", schedule);
            console.log("Schedule Type", scheduleType);
            console.log("Python Code:", pythonCode);

            return res.status(200).json({
                valid: true,
                message: 'Python code is valid',
                code: pythonCode, // Return the Python code
                savedFile: gpuPythonFile, // Return the path to the saved file
                savedScheduleFile: gpuPythonScheduleFile
            });
        }
    });
});
app.post('/task', (req, res) => {
    // Access the incoming request body using req.body
    console.log(req.body);

    // Perform any task or logic with the request data
    // For example, running a Python script, saving data, etc.

    // Send a response back
    res.json({ message: "Task received successfully" });
});

// Start the server
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
