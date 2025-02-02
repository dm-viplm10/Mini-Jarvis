import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  CircularProgress,
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import { Message } from '../types/api.ts';
import { FileUpload } from './FileUpload.tsx';
import ReactMarkdown from 'react-markdown';

interface ChatWindowProps {
  sessionId: string;
  messages: Message[];
  onSendMessage: (message: string, files: File[]) => Promise<void>;
  isLoading: boolean;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  sessionId,
  messages,
  onSendMessage,
  isLoading,
}) => {
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (message.trim() || files.length > 0) {
      await onSendMessage(message, files);
      setMessage('');
      setFiles([]);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Messages Area */}
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          p: 2,
          bgcolor: 'grey.50',
        }}
      >
        {messages.map((msg, index) => (
          <Box
            key={index}
            sx={{
              mb: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: msg.message.type === 'human' ? 'flex-end' : 'flex-start',
            }}
          >
            <Paper
              sx={{
                p: 2,
                maxWidth: '70%',
                bgcolor: msg.message.type === 'human' ? 'primary.light' : 'white',
                color: msg.message.type === 'human' ? 'white' : 'text.primary',
              }}
            >
              <ReactMarkdown>{msg.message.content}</ReactMarkdown>
              
              {/* Display attached files if any */}
              {msg.message.data?.files && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
                    Attached Files:
                  </Typography>
                  {msg.message.data.files.map((file, fileIndex) => (
                    <Typography key={fileIndex} variant="caption" display="block">
                      {file.name} ({(file.size / 1024).toFixed(1)} KB)
                    </Typography>
                  ))}
                </Box>
              )}

              {/* Display error if any */}
              {msg.message.data?.error && (
                <Typography color="error" sx={{ mt: 1 }}>
                  Error: {msg.message.data.error}
                </Typography>
              )}
            </Paper>
          </Box>
        ))}
        <div ref={messagesEndRef} />
      </Box>

      {/* File Upload Area */}
      <Box sx={{ p: 2, bgcolor: 'background.paper' }}>
        <FileUpload files={files} onFilesChange={setFiles} />
      </Box>

      {/* Input Area */}
      <Box
        sx={{
          p: 2,
          bgcolor: 'background.paper',
          borderTop: 1,
          borderColor: 'divider',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'flex-end' }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isLoading}
            sx={{ mr: 1 }}
          />
          <IconButton
            color="primary"
            onClick={handleSend}
            disabled={isLoading || (!message.trim() && files.length === 0)}
          >
            {isLoading ? <CircularProgress size={24} /> : <SendIcon />}
          </IconButton>
        </Box>
      </Box>
    </Box>
  );
};