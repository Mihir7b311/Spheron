import React from 'react';

const CodeNameInput = ({ codeName, setCodeName }) => {
  const handleNameChange = (e) => {
    setCodeName(e.target.value);
  };

  return (
    <div className="bg-[#1e1e2d] border border-[#2d2d3d] rounded-lg p-6">
      <label htmlFor="codeName" className="block text-sm font-medium text-gray-400 mb-2">
        Code Name
      </label>
      <input
        type="text"
        id="codeName"
        value={codeName}
        onChange={handleNameChange}
        placeholder="Enter a name for your code"
        className="w-full bg-[#2d2d3d] border border-[#3d3d4d] rounded-md px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
      />
    </div>
  );
};

export default CodeNameInput;