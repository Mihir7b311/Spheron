const cron = require('node-cron');
const fs = require('fs');
const path = require('path');
const axios = require('axios');

async function sendFilesToAPI(pythonFile,argFile){
    try {
        // Read file contents
        // const pythonScript = fs.readFileSync(pythonFile,'utf-8');
        // const argScript = fs.readFileSync(argFile,'utf-8');

        // Prepare the data to be sent to the API
        const requestData = {
            pythonScript: pythonFile,
            argScript: argFile
        }
        const response = await axios.post('http://localhost:5000/task',requestData);
        const response2 = await axios.post('http://0.0.0.0:8000/schedule',requestData);
        console.log('API Response:',response.data);
        console.log('Python API response :',response2);
    }
    catch (error) {
        console.log('Error sending files:',error);
    }
}







cron.schedule('* * * * *', () => {
    console.log("Cron job is running every minute to keep things active.");
});
// Function to handle daily schedules
console.log(new Date().toLocaleString());
function scheduleDailyTask(config,pythonFile,argFile) {
    const { startDate, time } = config;
    const [hour, minute] = time.split(':');
    const cronExpression = `${minute} ${hour} * * *`; // Run at the specified time every day

    const start = new Date(startDate);
    const now = new Date();

    if (start > now) {
        cron.schedule(cronExpression, () => {
            console.log(`Daily task running at ${hour}:${minute}`);
            sendFilesToAPI(pythonFile,argFile);
        });
    }
}

// Function to handle weekly schedules
function scheduleWeeklyTask(config,pythonFile,argFile) {
    const { startDate, time, days } = config;
    const [hour, minute] = time.split(':');

    const dayMapping = {
        Sun: 0,
        Mon: 1,
        Tue: 2,
        Wed: 3,
        Thu: 4,
        Fri: 5,
        Sat: 6
    };

    const cronDays = days.map(day => dayMapping[day]).join(',');

    const cronExpression = `${minute} ${hour} * * ${cronDays}`;

    const start = new Date(startDate);
    const now = new Date();

    if (start > now) {
        cron.schedule(cronExpression, () => {
            console.log(`Weekly task running at ${hour}:${minute}`);
            sendFilesToAPI(pythonFile,argFile);
        });
    }
}

// Function to handle monthly schedules
function scheduleMonthlyTask(config,pythonFile,argFile) {
    const { startDate, time } = config;
    const [hour, minute] = time.split(':');

    const cronExpression = `${minute} ${hour} 1 * *`; // Runs on the 1st day of every month at the specified time

    const start = new Date(startDate);
    const now = new Date();

    if (start > now) {
        cron.schedule(cronExpression, () => {

            console.log(`Monthly task running at ${hour}:${minute}`);
            sendFilesToAPI(pythonFile,argFile);
        });
    }
}

module.exports = { scheduleDailyTask, scheduleWeeklyTask, scheduleMonthlyTask };