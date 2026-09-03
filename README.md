# Agentic AI

A progressive series of Jupyter notebooks and a deployed application exploring the foundations of **Agentic AI** — from first API calls through multi-model orchestration, persona-grounded chatbots, tool-calling agents, and autonomous planning loops.

Built on **Python 3.12**, using NVIDIA's OpenAI-compatible API as the primary inference backend, with integrations across Gemini and Groq.

---

## Repository Structure

```
.
├── lab1.ipynb          # First LLM API calls & multi-step chaining
├── lab2.ipynb          # Multi-model competition & LLM-as-judge
├── lab3.ipynb          # Persona chatbot with evaluator & self-healing
├── lab4.ipynb          # Tool use, Pushover notifications & deployment
├── lab5.ipynb          # Tool use, LLM todos list generation & completion
├── app.py              # Production entry point — deployed to HuggingFace Spaces
├── linkedin/
│   ├── Profile.pdf    # LinkedIn profile (replace with your own)
│   └── summary.txt     # Personal summary (replace with your own)
└── .env                # API keys (not committed — see setup below)
```

---

## Labs Overview

### Lab 1 — First Steps with LLM APIs

**File:** `lab1.ipynb`

Sets up the development environment and makes the first API calls against **Gemini 2.5 Flash** via the OpenAI-compatible SDK. Covers environment variable loading with `python-dotenv`, client instantiation, and the OpenAI message format (`role` / `content`).

**Exercise:** Chains three sequential LLM calls to autonomously identify a business area, surface a pain point, and propose an Agentic AI solution — a preview of multi-step reasoning.

**Key concepts:** `load_dotenv`, `OpenAI` client, message history, `IPython.display.Markdown`

---

### Lab 2 — Multi-Model Competition & LLM-as-Judge

**File:** `lab2.ipynb`

Queries **five different LLMs** with the same challenging, open-ended question and then uses a sixth model as an autonomous judge to rank the responses. Introduces the **orchestrator pattern** — one model directing work across many others.

Models used: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `nvidia/nemotron-super-120b`, `gemini-2.5-flash`, `llama-3.3-70b-versatile` (Groq), `llama3.2` (local).

The judge returns a structured JSON ranking (`{"results": ["1","4","2","3","5"]}`), which is parsed and printed as a leaderboard.

**Key concepts:** Multi-provider API calls, parallelised prompting, structured JSON output, LLM-as-judge, orchestrator agentic pattern

---

### Lab 3 — Persona Chatbot with Self-Healing Evaluator

**File:** `lab3.ipynb`

Builds a **persona-grounded career chatbot** backed by a LinkedIn PDF and a `summary.txt`, served through a **Gradio** chat interface. The LLM responds in-character as the named individual.

Adds an **evaluator–rerun loop**: a second LLM (Gemini) grades every reply using a Pydantic `Evaluation` model (`is_acceptable: bool`, `feedback: str`). If a response fails, the system automatically retries with the rejection feedback injected into the system prompt — a basic self-healing agentic pattern, built without any framework.

**Key concepts:** `pypdf.PdfReader`, `gradio.ChatInterface`, `pydantic.BaseModel`, structured output parsing, evaluator pattern, quality-control loop

---

### Lab 4 — Tool Use, Push Notifications & HuggingFace Deployment

**File:** `lab4.ipynb` · `app.py`

The capstone lab. Extends the Lab 3 chatbot with **LLM Tool Use** — two callable functions the model can invoke autonomously:

| Tool                        | Trigger                       | Action                                                       |
| --------------------------- | ----------------------------- | ------------------------------------------------------------ |
| `record_user_details`     | Recruiter shares contact info | Logs name, email & notes; fires a Pushover push notification |
| `record_unknown_question` | Agent can't answer a question | Logs the question; fires a Pushover push notification        |

Implements the full **tool-call loop**: if `finish_reason == "tool_calls"`, execute the tools, append results as `role: "tool"` messages, and re-invoke the LLM for a final response.

`app.py` packages the complete agent as a standalone Gradio application and is deployed to **HuggingFace Spaces** via `uv run gradio deploy`.

**Key concepts:** JSON tool schemas, `handle_tool_calls()` dispatch, agentic loop, `requests.post` (Pushover), Gradio deployment, HuggingFace Spaces

---

### Lab 5 — Autonomous Agent Loop

**File:** `lab5.ipynb`

Explores **"The Unreasonable Effectiveness of the Agent Loop"** — building a fully autonomous planning agent from first principles using two todo-management tools:

| Tool              | Purpose                                                   |
| ----------------- | --------------------------------------------------------- |
| `create_todos`  | Accepts a list of step descriptions; initialises the plan |
| `mark_complete` | Marks a step done with completion notes                   |

Given a problem (e.g. a train-meeting word problem), the agent autonomously plans a step-by-step approach, executes each step in order using `globals().get(tool_name)` dispatch, marks tasks complete, and delivers a final answer — all within a `while finish_reason == "tool_calls"` loop. Progress is rendered live in the terminal using **Rich** console markup.

**Key concepts:** Tool schemas, `globals()` dispatch, `while` agent loop, `rich.console.Console`, autonomous multi-step reasoning

---

## Tech Stack

| Layer           | Libraries / Services                                                                                       |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| LLM Inference   | NVIDIA OpenAI-compatible API, Google Gemini API, Groq                                                      |
| Models          | `openai/gpt-oss-120b`, `gemini-2.5-flash`, `llama-3.3-70b-versatile`, `nvidia/nemotron-super-120b` |
| UI              | Gradio                                                                                                     |
| PDF Parsing     | pypdf                                                                                                      |
| Data Validation | Pydantic                                                                                                   |
| Notifications   | Pushover                                                                                                   |
| Formatting      | Rich                                                                                                       |
| Deployment      | HuggingFace Spaces (`uv run gradio deploy`)                                                              |
| Environment     | Python 3.12, uv, python-dotenv                                                                             |

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/Shakpro10/Agents.git
cd <your-repo>
uv sync
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# Primary inference (NVIDIA OpenAI-compatible)
NVIDIA_API_KEY_1=nvapi-...

# Secondary inference
NVIDIA_API_KEY_2=nvapi-...
GOOGLE_API_KEY=AIza...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...

# Push notifications (Lab 4 / app.py)
PUSHOVER_USER=u...
PUSHOVER_TOKEN=a...

# HuggingFace deployment (Lab 4)
HF_TOKEN=hf_...
```

### 3. Add your personal profile (Labs 3–5 / app.py)

Replace the files in the `linkedin/` directory:

```
linkedin/Profile.pdf     ← Export your LinkedIn profile as a PDF
linkedin/summary.txt      ← Write a short personal summary
```

Also update `self.name` in `app.py` to your own name.

---

## Deployment (Lab 4 / app.py)

Deploy the career chatbot to HuggingFace Spaces:

```bash
# Install the HuggingFace CLI
uv tool install 'huggingface_hub[cli]'

# Authenticate
hf auth login --token hf_xxx

# Deploy from the project root
uv run gradio deploy
```

When prompted: name the Space `career_conversation`, select `cpu-basic` hardware, and supply your API keys and Pushover secrets.

To update secrets after deployment, go to your Space → Settings → Variables and Secrets.

---

## Agentic Patterns Covered

| Pattern                                   | Lab        |
| ----------------------------------------- | ---------- |
| Sequential LLM chaining                   | Lab 1      |
| Orchestrator (one model directs many)     | Lab 2      |
| LLM-as-judge                              | Lab 2      |
| Evaluator + self-healing rerun loop       | Lab 3      |
| Tool use with real-world side effects     | Lab 4      |
| Autonomous tool-call agent loop           | Labs 4 & 5 |
| Autonomous planning with visible progress | Lab 5      |

---

## License

This project is for educational purposes. Replace all personal profile content in the `linkedin/` directory with your own before deploying.
