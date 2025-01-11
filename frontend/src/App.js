// App.js
import React, { useState } from 'react';
import Navbar from './components/Navbar';
import CodeBox from './components/CodeBox';
import PromptBox from './components/PromptBox';
import './styles/App.css';
import axios from 'axios';

const App = () => {
  const [code, setCode] = useState('');
  const [compileResult, setCompileResult] = useState(null);

  const handleCompile = async () => {
    try {
      const response = await axios.post('http://localhost:5000/check-python-syntax', { code });
      setCompileResult({ valid: true, message: response.data.message });
    } catch (error) {
      setCompileResult({
        valid: false,
        error: error.response?.data?.error || 'Unknown error occurred',
      });
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const fileContent = event.target.result;
        setCode(fileContent); // Set the uploaded file content in the CodeBox
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="app">
      <Navbar />
      <div className="main-content">
        <div className="left-panel">
          <CodeBox code={code} setCode={setCode} />
          <div className="button-container">
            <button className="upload-btn">
              Upload from Device
              <input
                type="file"
                className="file-input"
                accept=".py" // Accept only Python files
                onChange={handleFileUpload}
              />
            </button>
            <button className="compile-btn" onClick={handleCompile}>
              Compile
            </button>
          </div>
        </div>
        <div className="right-panel">
          <PromptBox />
          {compileResult && (
            <div className={`compile-result ${compileResult.valid ? 'success' : 'error'}`}>
              {compileResult.valid
                ? compileResult.message
                : `Error: ${compileResult.error}`}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
