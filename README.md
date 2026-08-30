# LangGraph Practice Lab

This repository is my learn-in-public trail through LangGraph and the surrounding LangChain ecosystem. It starts with small, explicit state machines and gradually grows into tool-using agents, persistent memory, multi-agent workflows, and self-correcting RAG systems.

The most useful shift in my mental model was this: an LLM call is only one step. A reliable application needs state, routing, persistence, observability, retrieval, validation, and clear stopping conditions. LangGraph gives me a way to make those decisions visible as a graph instead of hiding everything inside one large prompt.

## The progression

```text
Linear graphs
    -> conditional and parallel control flow
    -> conversational message state and streaming
    -> LangSmith traces and agent tool use
    -> RAG and human approval
    -> subgraphs and reusable composition
    -> short-term and long-term memory
    -> dynamic multi-agent fan-out
    -> corrective RAG and self-RAG verification loops
```

The numbered folders are chapters rather than production packages. Most experiments are notebooks so that each concept can be inspected cell-by-cell. The Python files in `05. langsmith/` and `08. mcp/` are the more script-like versions of the same learning process.

## What I understand now

### 1. A graph is an explicit state machine

Nodes are ordinary Python functions. Edges describe what happens next. The state is the contract between nodes. `START`, `END`, conditional routes, reducers, and checkpoints make the execution model inspectable and testable.

### 2. State design controls the quality of the workflow

Simple workflows use `TypedDict` fields such as `title`, `outline`, and `content`. Chat workflows use message lists with `add_messages`. Parallel workflows need reducers such as `operator.add` so that multiple branches can safely contribute to one state field. In the later RAG examples, the state becomes a deliberate audit trail: retrieved documents, relevance decisions, evidence, retry counts, and the final answer all have a place to live.

### 3. Routing is where an application becomes agentic

The conditional examples route based on structured model output, tool calls, relevance scores, support checks, and usefulness checks. This is more controlled than asking one model to improvise the entire workflow. It also creates natural places to add limits, fallbacks, logging, and tests.

### 4. Memory has two different jobs

Short-term memory is thread-scoped conversation state, handled here with `InMemorySaver` or `PostgresSaver`. Long-term memory is user-scoped information that can be reused across conversations, handled here with `InMemoryStore` or `PostgresStore`. The distinction matters: one preserves the conversation; the other preserves selected facts about the person.

### 5. Reliability comes from verification loops

The corrective RAG and self-RAG experiments treat retrieval and generation as hypotheses to evaluate. They score or filter retrieved context, check whether an answer is supported, revise unsupported answers, test whether the result is useful, and rewrite the retrieval query when necessary. Retry counters and `recursion_limit` are essential because a loop without a bound is a production incident waiting to happen.

## File-by-file learning map

The inventory below covers all 33 tracked project files. Each entry is based on the code and notebook cells currently in the repository.

### Configuration and foundations

| File | What it taught me |
| --- | --- |
| [.env.example](<.env.example>) | A single configuration surface is used for model providers, Tavily search, Gemini image generation, and LangSmith tracing. API keys and tracing settings belong in the environment rather than in graph logic. |

### `01. workflows/` — graph control flow

| File | What it taught me |
| --- | --- |
| [01. workflows/sequential.ipynb](<01. workflows/sequential.ipynb>) | A linear graph can model a dependable pipeline: `create_outline` writes to state, `create_blog` consumes that outline, and the graph ends. I understand nodes as state-transforming functions and compilation as the step that turns the builder into an executable workflow. |
| [01. workflows/parallel.ipynb](<01. workflows/parallel.ipynb>) | Three independent essay evaluators can run from `START`, then converge on `final_evaluation`. `Annotated[list[int], operator.add]` acts as a reducer so each branch contributes a score. This is the basic fan-out/fan-in pattern, plus structured output for bounded scores and feedback. |
| [01. workflows/conditional.ipynb](<01. workflows/conditional.ipynb>) | A sentiment classifier routes positive reviews directly to a thank-you response, while negative reviews go through diagnosis before support responds. Literal fields and Pydantic schemas keep route decisions and issue categories constrained. The key lesson is that the graph can branch on state produced by an earlier LLM call. |
| [01. workflows/iterative.ipynb](<01. workflows/iterative.ipynb>) | This chapter is the conceptual bridge from one-pass workflows to bounded improvement loops: produce an artifact, evaluate it, route back for another attempt when needed, and stop when a quality condition or retry budget is met. That same pattern appears later in corrective RAG and self-RAG. |

### `02. chatbot/` and `04. streaming/` — messages, threads, and user experience

| File | What it taught me |
| --- | --- |
| [02. chatbot/main.ipynb](<02. chatbot/main.ipynb>) | `MessagesState` plus the `add_messages` reducer gives the chatbot a durable conversation history. `MemorySaver` checkpoints that history, and a `thread_id` selects which conversation is being continued. I understand a checkpointer as the mechanism that turns a stateless graph invocation into a threaded conversation. |
| [04. streaming/main.ipynb](<04. streaming/main.ipynb>) | The same message graph can stream model messages through `chatbot.stream(..., stream_mode="messages")`. This separates graph execution from presentation: the model still runs in the node, while the caller progressively renders chunks for a responsive interface. |

### `05. langsmith/` — chains, observability, RAG, agents, and a real graph

| File | What it taught me |
| --- | --- |
| [05. langsmith/1_simple_llm_call.py](<05. langsmith/1_simple_llm_call.py>) | The smallest LangChain expression is a runnable pipeline: `PromptTemplate | ChatOpenAI | StrOutputParser`. The pipe operator composes stages, and `invoke` executes the complete chain. |
| [05. langsmith/2_sequential_chain.py](<05. langsmith/2_sequential_chain.py>) | A generated report can become the input to a second prompt that summarizes it. I learned how sequential runnable composition passes outputs between stages and how environment configuration controls LangSmith project metadata around a chain. |
| [05. langsmith/3_rag.py](<05. langsmith/3_rag.py>) | PDF RAG is a pipeline, not a single feature: load pages, split them with overlap, embed the chunks, index them in FAISS, retrieve the nearest chunks, format context, and ask the model to answer only from that context. `RunnableParallel` preserves the original question while retrieving context for the prompt. |
| [05. langsmith/3_rag_traceable_1.py](<05. langsmith/3_rag_traceable_1.py>) | `@traceable` makes setup functions such as PDF loading, splitting, and vector-store construction visible in LangSmith. A named query run makes the retrieval and answer stage easier to find and inspect. |
| [05. langsmith/3_rag_traceable_2.py](<05. langsmith/3_rag_traceable_2.py>) | Wrapping setup and query execution in a root traced function creates a parent-child run hierarchy. I understand the difference between tracing isolated helpers and tracing one complete request so latency, failures, and model calls can be understood in context. |
| [05. langsmith/3_rag_v4.py](<05. langsmith/3_rag_v4.py>) | A RAG index should be reusable. This version fingerprints the source file and indexing parameters, hashes that metadata into a cache key, saves FAISS locally, writes `meta.json`, and loads the cached index unless a rebuild is forced. Tags and metadata make setup, indexing, and QA runs queryable in LangSmith. |
| [05. langsmith/4_agent.py](<05. langsmith/4_agent.py>) | A ReAct agent combines an LLM with a web search tool and a custom weather tool. The model chooses whether to search or call weather, while `AgentExecutor` manages the loop and `max_iterations` provides a safety bound. Pulling a prompt from LangChain Hub also shows that agent behavior depends on the prompt/tool contract, not just the model. |
| [05. langsmith/5_langgraph.py](<05. langsmith/5_langgraph.py>) | This is a complete LangGraph evaluation workflow: language, analysis, and clarity evaluators run in parallel, then an aggregation node summarizes feedback and calculates the average. Structured essay scores, a reducer for parallel updates, LangSmith node traces, and invocation metadata all work together in one graph. |
| [05. langsmith/islr.pdf](<05. langsmith/islr.pdf>) | This bundled PDF is the reference corpus for the LangSmith PDF-RAG experiments. It makes the retrieval pipeline concrete: document loading, chunking, embedding, indexing, querying, tracing, and index reuse all operate on a local source document. |
| [05. langsmith/requirements.txt](<05. langsmith/requirements.txt>) | The pinned environment records the working dependency surface for LangChain, LangGraph, LangSmith, OpenAI, FAISS, PDF parsing, SQLite checkpointing support, Streamlit, and supporting libraries. Postgres and MCP-specific adapters are installed inline by the notebooks that need them. It is a snapshot for these experiments rather than a minimal production dependency file. |

### `08. mcp/` — external tools through MCP

| File | What it taught me |
| --- | --- |
| [08. mcp/main.py](<08. mcp/main.py>) | `MultiServerMCPClient` discovers tools from multiple MCP servers using different transports: local stdio for arithmetic and streamable HTTP for expenses. The graph binds those tools to the model, uses `ToolNode` to execute calls, routes with `tools_condition`, and loops back to the chat node until the model has a final answer. This is the tool lifecycle from discovery to execution inside an async graph. |

### `09. rag/` — retrieval as a graph tool and human approval

| File | What it taught me |
| --- | --- |
| [09. rag/basic.ipynb](<09. rag/basic.ipynb>) | A retriever can be exposed as a typed LangChain tool. The model decides when to call `rag_tool`, `ToolNode` executes it, and the graph returns to the model with retrieved context and metadata. This connects classic RAG with tool-calling control flow. |
| [09. rag/human_in_the_loop.ipynb.ipynb](<09. rag/human_in_the_loop.ipynb.ipynb>) | `interrupt` pauses a graph before a model response and returns a review payload to the caller. A checkpointer keeps the paused execution resumable, and `Command(resume=...)` continues the same thread after approval. Human-in-the-loop is therefore a first-class graph state transition, not an ad-hoc prompt. |

### `10. subgraphs/` — composing graphs

| File | What it taught me |
| --- | --- |
| [10. subgraphs/main.ipynb](<10. subgraphs/main.ipynb>) | A parent graph generates an English answer and hands the shared state to a compiled translation subgraph. The subgraph translates into Hindi and returns control to the parent. I understand subgraphs as reusable graph components that keep their own internal topology while participating in a larger workflow. |

### `11. memory/` — short-term state and long-term user memory

| File | What it taught me |
| --- | --- |
| [11. memory/short_term.ipynb](<11. memory/short_term.ipynb>) | `MessagesState` and `InMemorySaver` preserve a conversation under a `thread_id`. `get_state` exposes the latest checkpoint, making the stored message history inspectable. |
| [11. memory/stm_persistence.ipynb](<11. memory/stm_persistence.ipynb>) | `PostgresSaver` moves short-term checkpoints from process memory into Postgres. Thread 1 can remember a name across graph instances, while thread 2 remains isolated. This demonstrates durable state and conversation-level tenancy. |
| [11. memory/stm_deletion.ipynb](<11. memory/stm_deletion.ipynb>) | `RemoveMessage` lets a cleanup node delete old messages after a response. The graph keeps recent context bounded by removing the earliest six messages once the history grows beyond ten. |
| [11. memory/stm_trimming.ipynb](<11. memory/stm_trimming.ipynb>) | `trim_messages` selects the most recent messages that fit an approximate token budget before the model call. The graph can retain a fuller checkpoint while sending a smaller working context to the model, which is an important cost and context-window trade-off. |
| [11. memory/stm_summarization.ipynb](<11. memory/stm_summarization.ipynb>) | When the message history crosses a threshold, a summarizer compresses older turns into a `summary` field and removes all but the latest two messages. Future responses receive the summary as a system message. This is semantic compression rather than simple deletion. |
| [11. memory/ltm_basics.ipynb](<11. memory/ltm_basics.ipynb>) | `InMemoryStore` provides namespace/key/value long-term memory. The examples cover putting, getting, listing, and semantically searching user memories with embeddings. Namespaces scope facts to a user, while semantic search retrieves memories by meaning rather than exact wording. |
| [11. memory/ltm_implementation_chatbot.ipynb](<11. memory/ltm_implementation_chatbot.ipynb>) | This notebook evolves a memory-enabled chatbot in stages: read existing memories into a personalized system prompt; extract stable facts with structured output; write atomic records; reject duplicates with an `is_new` decision; and finally compose a `remember -> chat` graph. Store injection through `BaseStore` and `user_id` in runtime config are the key integration patterns. |
| [11. memory/ltm_postgres.ipynb](<11. memory/ltm_postgres.ipynb>) | `PostgresStore` provides durable long-term memories while keeping the same namespace and node design used with `InMemoryStore`. The notebook initializes the store, writes user facts, starts a fresh store context, and reads the facts back to verify persistence. |
| [11. memory/docker-compose.yml](<11. memory/docker-compose.yml>) | A local Postgres 16 service exposes port `5442` for the persistent memory exercises. It is the infrastructure layer behind both `PostgresSaver` and `PostgresStore`. The default credentials are suitable only for local learning. |

### `12. multi_agent_system/` — planning, fan-out, and reduction

| File | What it taught me |
| --- | --- |
| [12. multi_agent_system/blog_writing_agent.ipynb](<12. multi_agent_system/blog_writing_agent.ipynb>) | This is the largest workflow in the repository. A router decides whether research is needed, Tavily results are normalized into evidence, an orchestrator produces a structured plan, `Send` fans tasks out to section-writing workers, and a reducer subgraph merges sections. The reducer can also decide whether technical diagrams are useful, generate image assets with Gemini, place them into Markdown, and write the final article. The design demonstrates specialist workers, shared state, reducers, nested graphs, source grounding, and graceful image-generation fallback in one application-shaped example. |

### `13. corrective_rag/` and `14. self_rag/` — increasingly reliable RAG

| File | What it taught me |
| --- | --- |
| [13. corrective_rag/main.ipynb](<13. corrective_rag/main.ipynb>) | Retrieved chunks are evaluated one by one with a relevance score. The graph distinguishes `CORRECT`, `INCORRECT`, and `AMBIGUOUS`: strong internal evidence stays internal, weak retrieval triggers query rewriting and Tavily web search, and ambiguity combines both sources. Sentence decomposition plus an LLM filter creates a smaller refined context before generation. |
| [14. self_rag/main.ipynb](<14. self_rag/main.ipynb>) | The graph first decides whether retrieval is needed at all. Retrieved documents are filtered for topical relevance, the answer is generated from context, `IsSUP` checks support and captures evidence, and unsupported answers are revised into quote-only responses. `IsUSE` checks whether the answer actually answers the question; if not, the retrieval query is rewritten and the process retries. This separates retrieval necessity, relevance, grounding, usefulness, and termination into explicit decisions. |

## How to run the experiments

Create a virtual environment, install the shared dependencies, and configure the providers used by the notebook or script you want to run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "05. langsmith/requirements.txt"
cp .env.example .env
```

Then open the relevant notebook from the repository root:

```bash
jupyter lab
```

For the Postgres memory experiments:

```bash
docker compose -f "11. memory/docker-compose.yml" up -d
```

The notebooks intentionally use different local corpora, including paths such as `intro-to-ml.pdf`, `./documents/book1.pdf`, and `./documents/Company_Policies.pdf`. Those source documents are not all included in this repository, so the corresponding paths need to be supplied or changed before running those cells.

Some chapters also install or import chapter-specific packages inline, such as the Postgres checkpoint/store adapters, MCP adapters, and Gemini’s image SDK. The examples are learning artifacts tied to the library versions available when they were written, so small API adjustments may be needed when running them against newer releases.

## Engineering notes I am carrying forward

- Keep API keys in environment variables. The weather-tool experiment should be treated as a reminder to move provider credentials into configuration and rotate any credential that has ever been exposed in source.
- Keep external integrations configurable. The MCP example currently contains a machine-specific stdio path and a hosted server URL; a reusable application should load these from config.
- Treat Postgres defaults and `sslmode=disable` as local-development settings only.
- Treat indexes as build artifacts with a fingerprint, metadata, and an explicit invalidation strategy. The v4 RAG experiment is the first place this becomes intentional.
- Give every loop a budget. The self-RAG workflow uses retry counters and a recursion limit because correctness checks can otherwise become unbounded.
- Trace the whole request, not only the final model call. Parent-child runs make it possible to understand whether time and cost came from retrieval, embeddings, routing, generation, or retries.
- The numbered directories `03. persistence`, `06. observability`, and `07. tools` are directory-only chapter markers. Their ideas are developed in later artifacts: persistence in `11. memory`, observability in `05. langsmith`, and tools in `08. mcp` and `09. rag`.
- `myvenv/` is an ignored local virtual environment, including its generated package files; it is not part of the tracked learning material.

## The takeaway

I started by learning how to connect nodes. I ended up learning how to design controlled AI systems: define the state, make decisions explicit, isolate responsibilities, preserve the right memory, observe every step, verify the result, and fail safely when the evidence is not good enough. That is the part of LangGraph that feels most transferable beyond these exercises.
