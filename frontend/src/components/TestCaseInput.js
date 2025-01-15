import React, { useState } from 'react';
import { Play, Plus, Trash2 } from 'lucide-react';

const TestCaseInput = ({ onRun }) => {
  const [testCases, setTestCases] = useState([{ id: 1, value: '' }]);

  const addTestCase = () => {
    const newId = testCases.length + 1;
    setTestCases([...testCases, { id: newId, value: '' }]);
  };

  const removeTestCase = (id) => {
    if (testCases.length > 1) {
      setTestCases(testCases.filter(tc => tc.id !== id));
    }
  };

  const updateTestCase = (id, value) => {
    setTestCases(testCases.map(tc => 
      tc.id === id ? { ...tc, value } : tc
    ));
  };

  const handleRun = () => {
    if (onRun) {
      onRun(testCases.map(tc => tc.value));
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#1e1e2d] border border-[#2d2d3d] rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 bg-[#252533] border-b border-[#2d2d3d]">
        <span className="text-white font-semibold">Test Cases</span>
        <button 
          onClick={addTestCase}
          className="p-1 hover:bg-[#2d2d3d] rounded-md transition-colors"
        >
          <Plus size={16} className="text-[#6366f1]" />
        </button>
      </div>
      
      <div className="flex-1 p-4 overflow-y-auto">
        {testCases.map((testCase) => (
          <div key={testCase.id} className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Test Case {testCase.id}</span>
              {testCases.length > 1 && (
                <button
                  onClick={() => removeTestCase(testCase.id)}
                  className="p-1 hover:bg-[#2d2d3d] rounded-md transition-colors"
                >
                  <Trash2 size={14} className="text-red-400" />
                </button>
              )}
            </div>
            <textarea
              value={testCase.value}
              onChange={(e) => updateTestCase(testCase.id, e.target.value)}
              className="w-full h-24 bg-[#252533] border border-[#2d2d3d] rounded-md p-3 text-white text-sm font-mono resize-none focus:outline-none focus:border-[#6366f1]"
              placeholder="Enter test case..."
            />
          </div>
        ))}
      </div>
      
      <div className="px-4 py-3 bg-[#252533] border-t border-[#2d2d3d]">
        <button
          onClick={handleRun}
          className="w-full flex items-center justify-center gap-2 bg-[#6366f1] hover:bg-[#5355d1] text-white py-2 px-4 rounded-md transition-colors"
        >
          <Play size={16} />
          <span>Run Test Cases</span>
        </button>
      </div>
    </div>
  );
};

export default TestCaseInput;