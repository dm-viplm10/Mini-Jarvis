import axios from 'axios';
import { AgentRequest, AgentResponse } from '../types/api';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_TOKEN = process.env.REACT_APP_API_BEARER_TOKEN;

console.log(API_URL);
console.log(API_TOKEN);

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Authorization': `Bearer ${API_TOKEN}`,
    'Content-Type': 'application/json',
  },
});

export const sendAgentRequest = async (request: AgentRequest): Promise<void> => {
  await api.post('/api/mini-jarvis', request);
};

export const connectWebSocket = (sessionId: string, onMessage: (data: any) => void) => {
  const ws = new WebSocket(`ws://${API_URL.replace('http://', '')}/ws/${sessionId}`);
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };

  return ws;
}; 