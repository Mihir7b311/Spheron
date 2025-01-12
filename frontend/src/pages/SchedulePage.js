// pages/SchedulePage.js
import React, { useState, useEffect } from 'react';
import { Calendar, Clock, Code, ArrowRight } from 'lucide-react';
import Terminal from '../components/Terminal';

const SchedulePage = () => {
  const schedules = [
    {
      id: 1,
      name: "Daily Code Backup",
      type: "daily",
      time: "23:00",
      code: "import shutil\nfrom datetime import datetime\n\ndef backup_code():\n    backup_dir = f'backup_{datetime.now().strftime(\"%Y%m%d\")}'\n    shutil.make_archive(backup_dir, 'zip', './src')",
      lastRun: "2024-01-12",
      nextRun: "2024-01-13",
      status: "active",
      logMessages: [
        'Starting daily backup process...',
        'Scanning source directory...',
        'Found 156 files to backup',
        'Creating zip archive...',
        'Backup completed: backup_20240112.zip (45MB)',
        'Cleaning up temporary files...',
        'Daily backup successful'
      ]
    },
    {
      id: 2,
      name: "Weekly Report Generation",
      type: "weekly",
      days: ["Mon", "Thu"],
      time: "09:00",
      code: "import pandas as pd\n\ndef generate_report():\n    df = pd.read_csv('data.csv')\n    weekly_summary = df.groupby('category').sum()\n    weekly_summary.to_excel('weekly_report.xlsx')",
      lastRun: "2024-01-11",
      nextRun: "2024-01-15",
      status: "active",
      logMessages: [
        'Initializing weekly report generation...',
        'Loading data from data.csv',
        'Processing sales data...',
        'Calculating weekly metrics',
        'Generating Excel report...',
        'Sending email notifications',
        'Weekly report completed'
      ]
    },
    {
      id: 3,
      name: "Monthly Data Cleanup",
      type: "monthly",
      day: 1,
      time: "01:00",
      code: "def cleanup_data():\n    old_files = get_files_older_than(30)\n    for file in old_files:\n        archive_file(file)",
      lastRun: "2024-01-01",
      nextRun: "2024-02-01",
      status: "paused",
      logMessages: [
        'Starting monthly cleanup routine...',
        'Scanning for files older than 30 days',
        'Found 23 files for archival',
        'Compressing old files...',
        'Moving to archive storage',
        'Updating database records',
        'Monthly cleanup completed'
      ]
    }
  ];

  const [selectedSchedule, setSelectedSchedule] = useState(schedules[0] || null);
  const [terminalLogs, setTerminalLogs] = useState([]);

  // Reset and start new logs when selected schedule changes
  useEffect(() => {
    // Safety check - if no schedule is selected, don't proceed
    if (!selectedSchedule) return;

    let logIndex = 0;
    let intervalId = null;
    
    // Clear existing logs
    setTerminalLogs([]);

    // Add initial log
    setTerminalLogs([{
      timestamp: new Date().toLocaleTimeString(),
      type: 'info',
      message: `Initializing ${selectedSchedule.name}...`
    }]);

    const getLogType = () => {
      const rand = Math.random();
      if (rand < 0.6) return 'info';
      if (rand < 0.8) return 'success';
      if (rand < 0.9) return 'warning';
      return 'error';
    };

    // Ensure logMessages exists before setting up interval
    const messages = selectedSchedule.logMessages || [];

    intervalId = setInterval(() => {
      if (logIndex < messages.length) {
        setTerminalLogs(prevLogs => [
          ...prevLogs,
          {
            timestamp: new Date().toLocaleTimeString(),
            type: getLogType(),
            message: messages[logIndex]
          }
        ]);
        logIndex++;
      } else {
        // Add completion message
        setTerminalLogs(prevLogs => [
          ...prevLogs,
          {
            timestamp: new Date().toLocaleTimeString(),
            type: 'success',
            message: `${selectedSchedule.name} execution completed.`
          }
        ]);
        // Clear the interval
        if (intervalId) {
          clearInterval(intervalId);
        }
      }
    }, 2000);

    // Cleanup function
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [selectedSchedule]);

  // Safety check for rendering
  if (!schedules || schedules.length === 0) {
    return <div className="text-white p-4">No schedules available.</div>;
  }

  return (
    <div className="main-content" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', padding: '1rem' }}>
      {/* Left Panel - Schedule List */}
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-bold text-white mb-2">My Schedules</h1>
        <div className="flex flex-col gap-2 overflow-y-auto">
          {schedules.map((schedule) => (
            <div
              key={schedule.id}
              onClick={() => setSelectedSchedule(schedule)}
              className={`bg-[#1e1e2d] border border-[#2d2d3d] rounded-lg p-4 cursor-pointer transition-all hover:border-[#6366f1] ${
                selectedSchedule?.id === schedule.id ? 'border-[#6366f1]' : ''
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-white font-semibold">{schedule.name}</h3>
                <span className={`px-2 py-1 rounded text-xs ${
                  schedule.status === 'active' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
                }`}>
                  {schedule.status}
                </span>
              </div>
              <div className="flex items-center gap-4 text-gray-400 text-sm">
                <div className="flex items-center gap-1">
                  <Calendar size={14} />
                  <span>{schedule.type}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Clock size={14} />
                  <span>{schedule.time}</span>
                </div>
                <ArrowRight size={14} className="text-gray-600" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Middle Panel - Terminal */}
      <Terminal logs={terminalLogs} />

      {/* Right Panel - Schedule Details */}
      <div className="overflow-y-auto">
        {selectedSchedule && (
          <div className="bg-[#1e1e2d] border border-[#2d2d3d] rounded-lg p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold text-white">{selectedSchedule.name}</h2>
              <span className={`px-3 py-1 rounded-full text-sm ${
                selectedSchedule.status === 'active' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                {selectedSchedule.status}
              </span>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-[#252533] p-4 rounded-lg">
                <p className="text-gray-400 text-sm mb-1">Last Run</p>
                <p className="text-white">{selectedSchedule.lastRun}</p>
              </div>
              <div className="bg-[#252533] p-4 rounded-lg">
                <p className="text-gray-400 text-sm mb-1">Next Run</p>
                <p className="text-white">{selectedSchedule.nextRun}</p>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                <Code size={16} />
                Schedule Code
              </h3>
              <div className="bg-[#252533] p-4 rounded-lg">
                <pre className="text-gray-300 text-sm font-mono whitespace-pre-wrap">
                  {selectedSchedule.code}
                </pre>
              </div>
            </div>

            <div className="flex gap-3">
              <button className="px-4 py-2 bg-[#6366f1] text-white rounded-lg hover:bg-[#5355d1] transition-colors">
                Edit Schedule
              </button>
              {selectedSchedule.status === 'active' ? (
                <button className="px-4 py-2 bg-yellow-500/20 text-yellow-400 rounded-lg hover:bg-yellow-500/30 transition-colors">
                  Pause Schedule
                </button>
              ) : (
                <button className="px-4 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 transition-colors">
                  Resume Schedule
                </button>
              )}
              <button className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors">
                Delete Schedule
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SchedulePage;