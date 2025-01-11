import React from 'react';
import '../styles/PromptBox.css';

const PromptBox = () => {
  return (
    <div className="prompt-box-container">
      <textarea
        className="prompt-box"
        placeholder="Enter your prompt here..."
      />
    </div>
  );
};

export default PromptBox;