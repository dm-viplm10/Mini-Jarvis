# Mini Jarvis - Intelligent Agent-based Task Automation System [Using Pydantic AI]

## Repository Overview

Mini Jarvis is a sophisticated agent-based automation system that leverages multiple specialized AI agents to handle complex tasks. The system uses a director agent to orchestrate workflows and delegate tasks to specialized agents including data scientists, data engineers, web search, and GitHub integration capabilities.

### Key Features
- Multi-agent architecture with specialized roles
- FastAPI-based REST API interface
- Supabase integration for conversation memory
- Support for file processing and image analysis
- Secure authentication system
- Comprehensive data analysis and visualization capabilities

### Architecture
The system is built with a modular architecture consisting of:
- **Core Components**
  - FastAPI web server
  - Agent orchestration system
  - Memory management with Supabase
  - File processing system
- **Agent Types**
  - Director Agent: Orchestrates workflows
  - Data Scientist Agent: Handles data analysis and modeling
  - Data Engineer Agent: Manages data processing and pipelines
  - Data Team Agent: Coordinates data-related tasks
  - GitHub Agent: Handles version control operations
  - Web Search Agent: Performs internet research

## Installation & Setup Guide

### Prerequisites
- Python 3.10+
- PostgreSQL (for Supabase)
- Docker (optional)

### Dependencies
The project requires several Python packages listed in `requirements.txt`, including:
- FastAPI and Uvicorn for API server
- PyTorch for machine learning
- LlamaIndex for embeddings and vector stores
- Pandas and Scikit-learn for data processing
- OpenAI for language models
- Various other utilities and tools

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mini_jarvis.git
   cd mini_jarvis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file with the following variables:
   ```
   API_BEARER_TOKEN=your_api_token
   OPENROUTER_API_KEY=your_openrouter_key
   OPENROUTER_REFERER=your_referer
   OPENROUTER_TITLE=your_title
   ```

### Running the Project
1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. Using Docker:
   ```bash
   docker build -t mini-jarvis .
   docker run -p 8000:8000 mini-jarvis
   ```

## Usage Guide

### API Endpoints
- POST `/api/mini-jarvis`
  - Main endpoint for interacting with the system
  - Requires authentication via bearer token
  - Accepts JSON payload with:
    - `session_id`: Unique session identifier
    - `request_id`: Request identifier
    - `query`: User's query/task description
    - `files`: Optional array of files/images

### Example Request
```json
{
  "session_id": "user123",
  "request_id": "req456",
  "query": "Analyze the sales data and create a visualization",
  "files": []
}
```

## Code Documentation

### Main Components

#### Agent System (`agents/`)
- `director_agent.py`: Orchestrates workflow and delegates tasks
- `data_scientist_agent.py`: Handles data analysis and ML tasks
- `data_engineer_agent.py`: Manages data processing
- `data_team_agent.py`: Coordinates data operations
- `github_agent.py`: Handles version control tasks
- `web_search_agent.py`: Performs web searches

🎯 Director Agent
### Core Functions
- Workflow Planning & Execution
- Agent Task Distribution
- Context Management
- Response Validation

### Special Features
✨ Dynamic JSON Workflow Generation
✨ Real-time Agent Selection
✨ Cross-agent State Management
✨ Built-in Error Recovery

📊 Data Team Manager Agent
### Core Functions
- Project Structure Management
- Resource Allocation
- Quality Control
- Workflow Tracking

### Special Features
✨ Auto Project Organization
✨ Version Control Integration
✨ Cross-agent Coordination
✨ Smart File Management

🔬 Data Scientist Agent
### Core Functions
- ML Model Development
- Statistical Analysis
- Visualization Generation
- Performance Metrics

### Special Features
✨ Automated Feature Importance
✨ Smart Seasonal Analysis
✨ Intelligent Chart Selection
✨ Partial Data Analysis Support

⚙️ Data Engineer Agent
### Core Functions
- Data Validation
- Pipeline Management
- Data Transformation
- Quality Assurance

### Special Features
✨ Auto Quality Reports
✨ Smart Data Cleansing
✨ Multi-format Processing
✨ Pipeline Auto-optimization

🌐 Web Search Agent
### Core Functions
- Web Research
- Content Synthesis
- Source Verification
- Information Retrieval

### Special Features
✨ Semantic Search
✨ Relevance Scoring
✨ Real-time Updates
✨ Source Credibility Check

📂 GitHub Agent
### Core Functions
- Repository Management
- PR Processing
- Code Review
- Commit Analysis

### Special Features
✨ Auto PR Description
✨ Smart Diff Analysis
✨ Commit Pattern Recognition
✨ Repository Structure Analysis

🔄 Cross-Agent Capabilities
### Data Management
- Smart Data Routing
- Context Preservation
- Format Translation
- State Management

### Memory System
- Vector Store Integration
- Session Tracking
- Conversation Memory
- Context Retrieval

### Image Processing
- GPT-4 Vision Analysis
- Auto Visualization
- Multi-format Support
- Image Understanding

### Security
- Token Management
- Rate Limiting
- Access Control
- Secure API Handling


#### Core Functionality (`main.py`)
- FastAPI application setup
- Authentication middleware
- Workflow execution logic
- File processing utilities

#### Data Management
- Supabase integration for persistent storage
- File processing and image analysis capabilities
- Conversation history management

## Contribution Guidelines

### Development Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes
- Write unit tests for new features

### Testing
- Run tests using pytest
- Ensure all existing tests pass
- Add new tests for new functionality

## FAQ & Troubleshooting

### Common Issues
1. **Authentication Errors**
   - Verify API_BEARER_TOKEN is set correctly
   - Check token format in requests

2. **File Processing Issues**
   - Ensure file formats are supported
   - Check file size limits

3. **Agent Execution Errors**
   - Verify all required environment variables
   - Check agent dependencies

### Debugging Tips
- Enable debug logging
- Check application logs
- Verify API responses

## License

This project is licensed under the terms included in the LICENSE file.

## Support

For issues and feature requests, please use the github issue tracker.
