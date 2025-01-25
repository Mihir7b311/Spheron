const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const jwt = require('jsonwebtoken');

const pinataRoutes = require('./pinata'); 
const userRoutes = require('./user');
const db_then_ipfsRoutes = require('./upload_python_code');
require('./scheduleConfiguration.js');


const { scheduleDailyTask, scheduleWeeklyTask, scheduleMonthlyTask } = require('./scheduleConfiguration');

const app = express();
const SECRET_KEY = 'your_secret_key';

const corsOptions = {
    origin: '*',
    methods: ['GET', 'POST', 'PUT'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Authentication middleware
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) return res.status(401).json({ error: 'No token provided' });

    jwt.verify(token, SECRET_KEY, (err, user) => {
        if (err) return res.status(403).json({ error: 'Invalid token' });
        req.user = user;
        next();
    });
};

app.use('/pinata', pinataRoutes);
app.use('/api/auth', userRoutes);
app.use('/page',db_then_ipfsRoutes);

const TEMP_FOLDER = path.join(__dirname, 'temp');
const GPU_PYTHON_FOLDER = path.join(__dirname, 'GPUPython');

if (!fs.existsSync(TEMP_FOLDER)) fs.mkdirSync(TEMP_FOLDER, { recursive: true });
if (!fs.existsSync(GPU_PYTHON_FOLDER)) fs.mkdirSync(GPU_PYTHON_FOLDER, { recursive: true });

let fileCounter = 0;

// Protected routes
app.post('/check-python-syntax', authenticateToken, (req, res) => {
    const { code: pythonCode, schedule, userId } = req.body;

    if (!pythonCode) return res.status(400).json({ error: 'No Python code provided' });

    const tempFile = path.join(TEMP_FOLDER, `temp_script_${userId}.py`);
    fs.writeFileSync(tempFile, pythonCode);

    fileCounter += 1;
    const gpuPythonFile = path.join(GPU_PYTHON_FOLDER, `code${fileCounter}_${userId}.py`);
    const gpuPythonScheduleFile = path.join(GPU_PYTHON_FOLDER, `code${fileCounter}_${userId}.json`);
    
    fs.writeFileSync(gpuPythonFile, pythonCode);
    fs.writeFileSync(gpuPythonScheduleFile, JSON.stringify(schedule, null, 2));

    exec(`python -m py_compile ${tempFile}`, (error, stdout, stderr) => {
        fs.unlinkSync(tempFile);

        if (error) {
            return res.status(400).json({
                valid: false,
                error: stderr.trim(),
            });
        }

        if (schedule?.type) {
            const schedulers = {
                'daily': scheduleDailyTask,
                'monthly': scheduleMonthlyTask,
                'weekly': scheduleWeeklyTask
            };
            
            const scheduler = schedulers[schedule.type];
            if (scheduler) {
                scheduler(schedule.configuration, pythonCode, JSON.stringify(schedule, null, 2));
            }
        }

        return res.status(200).json({
            valid: true,
            message: 'Python code is valid',
            code: pythonCode,
            savedFile: gpuPythonFile,
            savedScheduleFile: gpuPythonScheduleFile
        });
    });
});

app.post('/task', authenticateToken, (req, res) => {
    console.log('Task from user:', req.user.id, req.body);
    res.json({ message: "Task received successfully" });
});

const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});