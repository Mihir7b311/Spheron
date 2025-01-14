const cron = require('node-cron');
cron.schedule('* * * * *', () => {
    console.log("Cron job is running every minute to keep things active.");
});
// Function to handle daily schedules
console.log(new Date().toLocaleString());
function scheduleDailyTask(config) {
    const { startDate, time } = config;
    const [hour, minute] = time.split(':');
    const cronExpression = `${minute} ${hour} * * *`; // Run at the specified time every day

    const start = new Date(startDate);
    const now = new Date();

    if (start > now) {
        cron.schedule(cronExpression, () => {
            console.log(`Daily task running at ${hour}:${minute}`);
        });
    }
}

// Function to handle weekly schedules
function scheduleWeeklyTask(config) {
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
        });
    }
}

// Function to handle monthly schedules
function scheduleMonthlyTask(config) {
    const { startDate, time } = config;
    const [hour, minute] = time.split(':');

    const cronExpression = `${minute} ${hour} 1 * *`; // Runs on the 1st day of every month at the specified time

    const start = new Date(startDate);
    const now = new Date();

    if (start > now) {
        cron.schedule(cronExpression, () => {
            console.log(`Monthly task running at ${hour}:${minute}`);
        });
    }
}

module.exports = { scheduleDailyTask, scheduleWeeklyTask, scheduleMonthlyTask };