export type AgentType = 'DIRECTOR' | 'GITHUB' | 'WEB_RESEARCH';

export interface UploadedFile {
  name: string;
  type: string;
  base64: string;
  size: number;
}

export interface AgentRequest {
  query: string;
  user_id: string;
  request_id: string;
  session_id: string;
  files?: UploadedFile[];
}

export interface AgentResponse {
  agent_type: AgentType;
  success: boolean;
  result: any;
  error?: string;
}

export interface Message {
  id: string;
  session_id: string;
  message: {
    type: 'human' | 'ai';
    content: string;
    data?: {
      files?: UploadedFile[];
      error?: string;
      request_id: string;
    };
  };
  created_at: string;
}

export interface Session {
  session_id: string;
  created_at: string;
} 