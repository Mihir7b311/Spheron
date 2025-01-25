import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import CodeBox from '../components/CodeBox';
import PromptBox from '../components/PromptBox';
import SchedulingSystem from '../components/SchedulingSystem';
import Terminal from '../components/Terminal';
import axiosInstance from '../utils/axios';
import { Play, Save } from 'lucide-react';

const DashboardPage = () => {
  const { user } = useAuth();
  const [code, setCode] = useState('');
  const [compileResult, setCompileResult] = useState(null);
  const [schedule, setSchedule] = useState({ data: null, displayText: '' });
  const [terminalLogs, setTerminalLogs] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  const handleUploadToDb = async () => {
    if (!code.trim()) {
      setUploadStatus({ type: 'error', message: 'No code to upload' });
      return;
    }

    try {
      const userId=user.id;
      const response = await axiosInstance.post(
        'http://localhost:5000/page/upload-to-db-then-ipfs',
        { code,userId,schedule } // Removed unnecessary `userId`
      );
      console.log("Schedule is:",schedule);

      setUploadStatus({
        type: 'success',
        message: `Code uploaded successfully! IPFS Hash: ${response.data.ipfsHash}`,
      });
    } catch (error) {
      setUploadStatus({
        type: 'error',
        message: error.response?.data?.error || 'Failed to upload code',
      });
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setCode(event.target.result); // Set file content
      };
      reader.readAsText(file);
    }
  };

  const handleCompile = async () => {
    try {
      const response = await axiosInstance.post('/check-python-syntax', { 
        code,
        schedule: schedule.data,
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
    setTerminalLogs([{ timestamp: new Date().toLocaleTimeString(), type: 'info', message: 'Starting code execution...' }]);

    try {
      const response = await axiosInstance.post('/run-code', { code });

      const executionLogs = response.data.logs || [
        { type: 'info', message: 'Setting up environment...' },
        { type: 'info', message: 'Executing Python script...' },
        { type: 'success', message: 'Script execution completed.' },
        { type: 'info', message: `Output: ${response.data.output}` },
      ];

      executionLogs.forEach((log, index) => {
        setTimeout(() => {
          setTerminalLogs((prevLogs) => [
            ...prevLogs,
            { timestamp: new Date().toLocaleTimeString(), type: log.type, message: log.message },
          ]);
        }, index * 500);
      });
    } catch (error) {
      setTerminalLogs((prevLogs) => [
        ...prevLogs,
        { timestamp: new Date().toLocaleTimeString(), type: 'error', message: `Error: ${error.response?.data?.error || 'Failed to execute code'}` },
      ]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleScheduleChange = async (scheduleData) => {
    setSchedule(scheduleData);
    try {
      await axiosInstance.post('/save-schedule', { schedule: scheduleData.data });
    } catch (error) {
      console.error('Failed to save schedule:', error);
    }
  };

  return (
    <div className="main-content" style={{ display: 'flex', gap: '1rem', padding: '1rem' }}>
      <div className="left-panel" style={{ flex: '0 0 65%' }}>
        <div className="flex flex-col gap-4">
          <CodeBox code={code} setCode={setCode} />

          <div className="button-container">
            <button className="upload-btn">
              Upload from Device
              <input type="file" className="file-input" accept=".py" onChange={handleFileUpload} />
            </button>
            <button
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2"
              onClick={handleUploadToDb}
            >
              <Save size={16} />
              Save to DB
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

          {uploadStatus && (
            <div
              className={`px-4 py-2 rounded-lg ${
                uploadStatus.type === 'success' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
              }`}
            >
              {uploadStatus.message}
            </div>
          )}

          <div className="h-52">
            <Terminal logs={terminalLogs} />
          </div>
        </div>
      </div>

      <div className="right-panel" style={{ flex: '0 0 35%' }}>
        <PromptBox />
        <div className="scheduling-wrapper mt-4">
          <div className="bg-[#1e1e2d] border border-[#2d2d3d] rounded-lg p-6">
            <SchedulingSystem onScheduleChange={handleScheduleChange} />
          </div>
        </div>

        {schedule.displayText && (
          <div className="schedule-prompt mt-4 px-4 py-2 bg-[#1e1e2d] border border-[#2d2d3d] rounded-lg">
            <p className="text-sm text-gray-400">Current Schedule:</p>
            <p className="text-white">{schedule.displayText}</p>
          </div>
        )}

        {compileResult && (
          <div
            className={`compile-result mt-4 px-4 py-2 rounded-lg ${
              compileResult.valid ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}
          >
            {compileResult.valid ? compileResult.message : `Error: ${compileResult.error}`}
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardPage;
