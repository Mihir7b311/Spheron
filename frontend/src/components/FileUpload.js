import React, { useState } from 'react';
import { Upload, AlertCircle, CheckCircle2 } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import axiosInstance from '../utils/axios';

const FileUpload = ({ onCodeUpdate }) => {
  const [uploadStatus, setUploadStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus(null);

    try {
      // First, read the file content for the CodeBox
      const reader = new FileReader();
      reader.onload = (event) => {
        const fileContent = event.target.result;
        onCodeUpdate(fileContent);
      };
      reader.readAsText(file);

      // Then, upload to Pinata
      const formData = new FormData();
      formData.append('file', file);

      const response = await axiosInstance.post('/pinata/upload-python-code', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadStatus({
        type: 'success',
        message: 'File uploaded successfully! IPFS Hash: ' + response.data.ipfsHash
      });
    } catch (error) {
      setUploadStatus({
        type: 'error',
        message: error.response?.data?.error || 'Failed to upload file'
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full space-y-4">
      <div className="flex items-center gap-4">
        <label className="relative flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors cursor-pointer">
          <Upload size={16} />
          {isUploading ? 'Uploading...' : 'Upload Python File'}
          <input
            type="file"
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            accept=".py"
            onChange={handleFileUpload}
            disabled={isUploading}
          />
        </label>
      </div>

      {uploadStatus && (
        <Alert variant={uploadStatus.type === 'success' ? 'default' : 'destructive'}>
          {uploadStatus.type === 'success' ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          <AlertDescription>{uploadStatus.message}</AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default FileUpload;