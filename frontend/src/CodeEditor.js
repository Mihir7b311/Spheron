import React, { useState } from 'react';
import Editor from '@monaco-editor/react';
import { Play, Save, FileCode, Settings, Terminal } from 'lucide-react';

const PyIDE = () => {
  const [code, setCode] = useState('# Write your Python code here\n\ndef hello_world():\n    print("Hello, World!")');

  return (
    <div className="flex h-screen bg-gray-900">
      {/* Left Navbar */}
      <nav className="w-16 bg-gray-800 flex flex-col items-center py-4 border-r border-gray-700">
        <div className="space-y-6">
          <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">
            <FileCode size={24} />
          </button>
          <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">
            <Play size={24} />
          </button>
          <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">
            <Save size={24} />
          </button>
          <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">
            <Terminal size={24} />
          </button>
          <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">
            <Settings size={24} />
          </button>
        </div>
      </nav>

      {/* Main Editor Area */}
      <div className="flex" style={{ width: '50%' }}>
        <Editor
          height="100vh"
          defaultLanguage="python"
          theme="vs-dark"
          value={code}
          onChange={(value) => setCode(value)}
          options={{
            minimap: { enabled: true },
            fontSize: 14,
            lineNumbers: 'on',
            roundedSelection: false,
            scrollBeyondLastLine: false,
            readOnly: false,
            automaticLayout: true,
          }}
        />
      </div>

      {/* Right White Space */}
      <div className="flex-1 bg-white"></div>
    </div>
  );
};

export default PyIDE;
