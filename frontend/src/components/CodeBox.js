// CodeBox.js
import React from 'react';
import '../styles/CodeBox.css';

const CodeBox = ({ code, setCode }) => {
  const lines = code.split('\n');

  return (
    <div className="code-box-container">
      <div className="line-numbers">
        {lines.map((_, index) => (
          <div key={index + 1} className="line-number">
            {index + 1}
          </div>
        ))}
      </div>
      <textarea
        className="code-box"
        value={code}
        onChange={(e) => setCode(e.target.value)}
        placeholder="Enter your code here..."
        spellCheck="false"
      />
    </div>
  );
};

export default CodeBox;
