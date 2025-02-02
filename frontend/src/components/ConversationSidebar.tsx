import React from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Typography,
  Divider,
  IconButton,
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import { Session } from '../types/api';

interface ConversationSidebarProps {
  sessions: Session[];
  currentSessionId: string;
  onSessionSelect: (sessionId: string) => void;
  onNewSession: () => void;
}

export const ConversationSidebar: React.FC<ConversationSidebarProps> = ({
  sessions,
  currentSessionId,
  onSessionSelect,
  onNewSession,
}) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Box
      sx={{
        width: 280,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRight: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Typography variant="h6">Conversations</Typography>
        <IconButton color="primary" onClick={onNewSession}>
          <AddIcon />
        </IconButton>
      </Box>
      
      <Divider />
      
      <List sx={{ flex: 1, overflowY: 'auto' }}>
        {sessions.map((session) => (
          <ListItem key={session.session_id} disablePadding>
            <ListItemButton
              selected={session.session_id === currentSessionId}
              onClick={() => onSessionSelect(session.session_id)}
            >
              <ListItemText
                primary={`Session ${session.session_id.slice(0, 8)}...`}
                secondary={formatDate(session.created_at)}
                primaryTypographyProps={{
                  noWrap: true,
                }}
                secondaryTypographyProps={{
                  noWrap: true,
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );
};