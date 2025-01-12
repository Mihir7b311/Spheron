// pages/DashboardPage.js
import React, { useState } from 'react';
import axios from 'axios';
import { Play } from 'lucide-react';
import CodeBox from '../components/CodeBox';
import PromptBox from '../components/PromptBox';
import SchedulingSystem from '../components/SchedulingSystem';
import Terminal from '../components/Terminal';

const DashboardPage = () => {
  const [code, setCode] = useState('');
  const [compileResult, setCompileResult] = useState(null);
  const [schedule, setSchedule] = useState({ data: null, displayText: '' });
  const [terminalLogs, setTerminalLogs] = useState([]);
  const [isRunning, setIsRunning] = useState(false);

  const handleCompile = async () => {
    try {
      const response = await axios.post('http://localhost:5000/check-python-syntax', { 
        code,
        schedule: schedule.data
      });
      setCompileResult({ valid: true, message: response.data.message });
    } catch (error) {
      setCompileResult({
        valid: false,
        error: error.response?.data?.error || 'Unknown error occurred',
      });
    }
  };

  const handleRunNow = async () => {
    setIsRunning(true);
    // Clear previous logs
    setTerminalLogs([{
      timestamp: new Date().toLocaleTimeString(),
      type: 'info',
      message: 'Starting code execution...'
    }]);

    try {
      // Make API call to run the code
      const response = await axios.post('http://localhost:5000/run-code', { 
        code,
      });

      // Simulate receiving logs from the server
      // In reality, this would be replaced with actual server response handling
      const executionLogs = response.data.logs || [
        { type: 'info', message: 'Setting up environment...' },
        { type: 'info', message: 'Executing Python script...' },
        { type: 'success', message: 'Script execution completed.' },
        { type: 'info', message: 'Output: ' + response.data.output }
      ];

      // Add each log with timestamp
      executionLogs.forEach((log, index) => {
        setTimeout(() => {
          setTerminalLogs(prevLogs => [...prevLogs, {
            timestamp: new Date().toLocaleTimeString(),
            type: log.type,
            message: log.message
          }]);
        }, index * 500);
      });

    } catch (error) {
      setTerminalLogs(prevLogs => [...prevLogs, {
        timestamp: new Date().toLocaleTimeString(),
        type: 'error',
        message: `Error: ${error.response?.data?.error || 'Failed to execute code'}`
      }]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const fileContent = event.target.result;
        setCode(fileContent);
      };
      reader.readAsText(file);
    }
  };

  const handleScheduleChange = (scheduleData) => {
    setSchedule(scheduleData);
  };

  return (
    <div className="main-content">
      <div className="left-panel">
        {/* Code Editor Section */}
        <div className="flex flex-col gap-4">
          <CodeBox code={code} setCode={setCode} />
          
          {/* Buttons row */}
          <div className="button-container">
            <button className="upload-btn">
              Upload from Device
              <input
                type="file"
                className="file-input"
                accept=".py"
                onChange={handleFileUpload}
              />
            </button>
            <button className="compile-btn" onClick={handleCompile}>
              Compile
            </button>
            <button 
              onClick={handleRunNow}
              disabled={isRunning}
              className={`px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2 ${
                isRunning ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              <Play size={16} />
              {isRunning ? 'Running...' : 'Run Now'}
            </button>
          </div>

          {/* Terminal Section */}
          <div className="h-64">
            <Terminal logs={terminalLogs} />
          </div>
        </div>
      </div>

      <div className="right-panel">
        <PromptBox />
        <div className="scheduling-wrapper">
          <SchedulingSystem onScheduleChange={handleScheduleChange} />
        </div>
        {/* Display current schedule */}
        {schedule.displayText && (
          <div className="schedule-prompt px-4 py-2 bg-gray-700 text-gray-300 rounded">
            <p className="text-sm">Current Schedule:</p>
            <p className="text-white">{schedule.displayText}</p>
          </div>
        )}
        {compileResult && (
          <div className={`compile-result ${compileResult.valid ? 'success' : 'error'}`}>
            {compileResult.valid
              ? compileResult.message
              : `Error: ${compileResult.error}`}
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardPage;