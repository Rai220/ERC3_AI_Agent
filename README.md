# ERC3 AI Agent

🏆 **7th place** in the main [Enterprise RAG Challenge 3: AI Agents](https://erc.timetoact-group.at/) competition!

Sample agents demonstrating how to participate in the ERC3: AI Agents in Action competition.

## Agent Architecture

The agent is built using **GPT-5.1** with a vanilla **ReAct agent** on **LangGraph**.

### Tools

All ERC functions are implemented as tools, plus a few additional tools following agent-building best practices:

- **plan tool** — for planning and breaking down complex tasks
- **think tool** — for controlled reasoning
- **critic tool** — uses structured output with dedicated reasoning fields

### Context Management

Context is a single continuous thread: at any moment the agent can see the full chain of its own reasoning and actions. Everything else was achieved through careful prompt engineering.

### Cost

One full agent run on GPT-5.1 costs approximately **$5** for 106 tasks.

## Getting Started

### 1. Get Your API Key

To use these agents, you'll need an ERC3 API key:

1. Visit https://erc.timetoact-group.at/
2. Enter the email address you used during registration
3. Your API key will be displayed

Note: If you haven't registered yet, https://www.timetoact-group.at/events/enterprise-rag-challenge-part-3 and allow 24 hours for your
registration to be processed.

### 2. Prerequisites

All agents require:
- ERC3 SDK - for connecting to the platform and accessing benchmarks
- ERC3_API_KEY - your competition API key
- LLM API Key - such as OPENAI_API_KEY or equivalent (depending on the agent)

## Running an Agent

Here's an example of running the sgr-agent-store (a simple agent that solves the store benchmark):

```bash
# Set up your environment variables
export OPENAI_API_KEY=sk-...
export ERC3_API_KEY=key-...

# Navigate to the agent directory
cd sgr-agent-store

# Activate your virtual environment (optional but recommended)
# python3 -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the agent
python3 main.py
```

# Available Agents

- sgr-agent-store - A simple agent implementation for [STORE benchmark](https://erc.timetoact-group.at/benchmarks/store). It relies on [Schema-Guided Reasoning](https://abdullin.com/schema-guided-reasoning/) to provide adaptive thinking capabilities with a single recursive prompt and gpt-4o.
- sgr-agent-erc3 - A simple [SGR](https://abdullin.com/schema-guided-reasoning/) NextStep agent for [ERC3-DEV benchmark](https://erc.timetoact-group.at/benchmarks/erc3-dev).

# Resources

- [Enterprise RAG Challenge 3](https://erc.timetoact-group.at/) — competition platform
- https://www.timetoact-group.at/events/enterprise-rag-challenge-part-3

# Support

You can ask questions in the discord channel (you get a link to that with the registration email)

# Author

📢 Follow my Telegram channel for more AI/ML content and source code updates: [RoboFuture](https://t.me/robofuture)
