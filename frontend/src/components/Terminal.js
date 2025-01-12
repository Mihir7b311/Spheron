// components/Terminal.js
import React, { useEffect, useRef } from 'react';
import { Terminal as TerminalIcon } from 'lucide-react';

const Terminal = ({ logs }) => {
  const terminalRef = useRef(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="h-full flex flex-col bg-[#1e1e2d] border border-[#2d2d3d] rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 bg-[#252533] border-b border-[#2d2d3d]">
        <TerminalIcon size={16} className="text-[#6366f1]" />
        <span className="text-white font-semibold">Execution Logs</span>
      </div>
      <div 
        ref={terminalRef}
        className="flex-1 p-4 font-mono text-sm overflow-y-auto"
      >
        {logs.map((log, index) => (
          <div key={index} className="mb-1">
            <span className="text-gray-500">[{log.timestamp}]</span>{" "}
            <span className={`
              ${log.type === 'error' ? 'text-red-400' : ''}
              ${log.type === 'success' ? 'text-green-400' : ''}
              ${log.type === 'info' ? 'text-blue-400' : ''}
              ${log.type === 'warning' ? 'text-yellow-400' : ''}
            `}>
              {log.message}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Terminal;