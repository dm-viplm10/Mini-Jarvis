import React, { useState, useEffect } from 'react';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { ChatWindow } from './components/ChatWindow.tsx';
import { Message, AgentRequest, AgentResponse, UploadedFile } from './types/api';
import { sendAgentRequest } from './services/api.ts';
import { createClient } from '@supabase/supabase-js';
import { v4 as uuidv4 } from 'uuid';

const supabase = createClient(
  process.env.REACT_APP_SUPABASE_URL || '',
  process.env.REACT_APP_SUPABASE_ANON_KEY || ''
);

const theme = createTheme({
  palette: {
    primary: { main: '#2196f3' },
    background: { default: '#f5f5f5' },
  },
});

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentSessionId] = useState<string>(uuidv4());
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Start a new session
    const startSession = async () => {
      try {
        await supabase
          .from('sessions')
          .insert([{ session_id: currentSessionId }]);
      } catch (error) {
        console.error('Error starting session:', error);
      }
    };
    startSession();

    // Subscribe to messages table for real-time updates
    const messagesChannel = supabase
      .channel('messages-channel')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'messages',
          filter: `session_id=eq.${currentSessionId}`,
        },
        (payload) => {
          const newMessage = payload.new as Message;
          setMessages((prev) => [...prev, newMessage]);
        }
      )
      .subscribe();

    // Load existing messages
    const loadMessages = async () => {
      try {
        const { data, error } = await supabase
          .from('messages')
          .select('*')
          .eq('session_id', currentSessionId)
          .order('created_at', { ascending: true });

        if (error) throw error;
        if (data) setMessages(data);
      } catch (error) {
        console.error('Error loading messages:', error);
      }
    };
    loadMessages();

    return () => {
      supabase.removeChannel(messagesChannel);
    };
  }, [currentSessionId]);

  const handleSendMessage = async (message: string, files: File[]) => {
    try {
      setIsLoading(true);

      // Convert File objects to UploadedFile format
      const uploadedFiles = await Promise.all(
        files.map(async (file) => {
          return new Promise<UploadedFile>((resolve) => {
            const reader = new FileReader();
            reader.onload = () => {
              const base64 = reader.result?.toString().split(',')[1] || '';
              resolve({
                name: file.name,
                type: file.type,
                size: file.size,
                base64
              });
            };
            reader.readAsDataURL(file);
          });
        })
      );

      const request: AgentRequest = {
        query: message,
        user_id: 'default-user',
        session_id: currentSessionId,
        request_id: uuidv4(),
        files: uploadedFiles,
      };

      // Store user message in Supabase
      await supabase.from('messages').insert([{
        session_id: currentSessionId,
        message: {
          type: 'human',
          content: message,
          data: {
            files: uploadedFiles,
            request_id: request.request_id,
          }
        }
      }]);

      // Send request to backend
      await sendAgentRequest(request);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Store error message in Supabase
      await supabase.from('messages').insert([{
        session_id: currentSessionId,
        message: {
          type: 'ai',
          content: 'Failed to send message. Please try again.',
          data: {
            error: error instanceof Error ? error.message : 'Unknown error',
            request_id: uuidv4(),
          }
        }
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', overflow: 'hidden' }}>
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <ChatWindow
            sessionId={currentSessionId}
            messages={messages}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
          />
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;