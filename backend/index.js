const cron = require('node-cron');

// Schedule a task to run every minute
cron.schedule('* * * * *', () => {
    console.log('Task is running every minute:', new Date());
});

console.log('Scheduler is running...');
