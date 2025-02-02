import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, List, ListItem, ListItemText, IconButton } from '@mui/material';
import { Delete as DeleteIcon, AttachFile as AttachFileIcon } from '@mui/icons-material';

interface FileUploadProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
}

const MAX_FILES = 5;
const ACCEPTED_TYPES = {
  'text/plain': ['.txt'],
  'application/json': ['.json'],
  'text/csv': ['.csv'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png']
};

export const FileUpload: React.FC<FileUploadProps> = ({ files, onFilesChange }) => {
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (files.length + acceptedFiles.length > MAX_FILES) {
      alert(`Maximum ${MAX_FILES} files allowed`);
      return;
    }
    onFilesChange([...files, ...acceptedFiles]);
  }, [files, onFilesChange]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: MAX_FILES - files.length
  });

  const removeFile = (index: number) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    onFilesChange(newFiles);
  };

  return (
    <Box>
      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          borderRadius: 1,
          p: 2,
          mb: 2,
          cursor: 'pointer',
          bgcolor: isDragActive ? 'action.hover' : 'background.paper',
        }}
      >
        <input {...getInputProps()} />
        <Box display="flex" alignItems="center" justifyContent="center">
          <AttachFileIcon sx={{ mr: 1 }} />
          <Typography>
            {isDragActive
              ? 'Drop files here'
              : `Drag files here or click to select (${files.length}/${MAX_FILES})`}
          </Typography>
        </Box>
      </Box>

      <List>
        {files.map((file, index) => (
          <ListItem
            key={index}
            secondaryAction={
              <IconButton edge="end" onClick={() => removeFile(index)}>
                <DeleteIcon />
              </IconButton>
            }
          >
            <ListItemText
              primary={file.name}
              secondary={`${(file.size / 1024).toFixed(1)} KB`}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );
};
