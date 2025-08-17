# agent_llm.py (migrated)
import uuid
import json
from typing import List, Tuple, Any

# LangChain / LangGraph message types
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages

# LangGraph memory and agent factory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Ollama chat model (your original)
from langchain_ollama.chat_models import ChatOllama

# Your tool definitions (assumed to be Tool-like objects with .name and .func)
from agent_tools import get_air_quality_tool, get_weather_tool, predict_aqi_tool, both_tool

# -------------------------
# Initialize memory and model
# -------------------------
memory = MemorySaver()
model = ChatOllama(
    model="llama3.2:latest",
    temperature=0.6,
    validate_model_on_init=True
)

# -------------------------
# System message for LLM
# -------------------------
system_message = SystemMessage(
    content="""
You are an environmental assistant. Depending on the user query, follow one of these workflows:

(1) air_quality_query:
 - Extract city_name, country_code, and cnt (for forecast only)
 - Get coordinates if needed
 - Call "get_air_quality_tool"
 - Output plot (base64) and textual summary

(2) weather_query:
 - Extract city_name, country_code, and cnt (for forecast only)
 - Call "get_weather_tool"
 - Output plots (weather, rain, wind in base64) and textual summary

(3) predict_aqi_query:
 - Extract pollutant values (pm2.5, pm10, CO, NO2, SO2, O3)
 - Call "predict_aqi_tool"
 - Output textual summary

(4) both_query:
 - Extract city_name, country_code, and cnt (for forecast only)
 - Execute "both_tool"
 - Combine results, plots, and summary

(5) general_query:
 - Use web search or document retrieval if needed
 - Generate textual summary

(6) mix:
 - Determine which tools are required
 - Execute corresponding workflows
 - Combine results

**Rules:**
 - ALWAYS output structured JSON for tool calls before any textual response. Example:
   {
     "name": "air_quality_workflow",
     "parameters": {
       "city_name": "Delhi",
       "country_code": "IN",
       "cnt": 7
     }
   }
 - Only output JSON for tool calls. Then, once the tool result is returned, provide a friendly textual summary to the user.
 - For plots, the tools return base64 images. Include them in the response if available.
"""
)

# -------------------------
# Tools list (Tool-like objects expected)
# -------------------------
tools = [get_air_quality_tool, get_weather_tool, predict_aqi_tool, both_tool]


# -------------------------
# Helper: conservative token counter used by trim_messages
# -------------------------
def simple_token_count(messages: List[Any]) -> int:
    # Very conservative/cheap token estimator: number of characters.
    return sum(len(getattr(m, "content", "") or "") for m in messages)


# -------------------------
# Prompt function (expected by create_react_agent)
# -------------------------
# safe prompt that accepts different 'state' shapes
def prompt(state):
    """
    Accepts a 'state' which can be:
      - a list of messages (HumanMessage/AIMessage/etc.),
      - a dict containing the key "messages" -> list,
      - an object with attribute .messages (list).
    Returns trimmed messages (including the system message).
    """
    # normalize to a list of messages
    if isinstance(state, list):
        state_messages = state
    elif isinstance(state, dict) and "messages" in state and isinstance(state["messages"], list):
        state_messages = state["messages"]
    elif hasattr(state, "messages"):
        try:
            state_messages = list(state.messages)
        except Exception:
            state_messages = []
    else:
        state_messages = []

    # now it's safe to concatenate
    messages = [system_message] + list(state_messages)

    return trim_messages(
        messages,
        token_counter=simple_token_count,
        max_tokens=2000,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False
    )



# -------------------------
# Create agent (prebuilt react-style)
# -------------------------
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=prompt,
    checkpointer=memory,
)


# -------------------------
# Utility: try-parse JSON block from model content robustly
# -------------------------
def extract_json_from_text(text: str):
    """
    Attempt to find the first JSON object in text and parse it.
    Returns dict on success or None.
    """
    text = text.strip()
    # If the whole content is JSON, parse directly
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        pass

    # Otherwise, find the first '{' ... '}' balanced JSON substring
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed
                except Exception:
                    return None
    return None


# -------------------------
# Option A — simple: let the agent execute tools (preferred if agent.run supports it)
# -------------------------

def run_agent_simple(user_query: str, session_state_messages: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Delegates tool execution to the agent runtime (if available).
    Returns (outputs, final_state_messages).
    """
    # Build message thread for input
    state_messages = list(session_state_messages) + [HumanMessage(content=user_query)]
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Agent.run commonly exists on prebuilt agents. It may either return final message text or list.
    # If your version of create_react_agent supports execute_tools via run, this is the simplest.
    result = agent.run({"messages": state_messages}, config)  # NOTE: some versions accept execute_tools arg
    outputs = [result] if result is not None else []

    # Append LLM result to state_messages for continuity
    state_messages.append(AIMessage(content=str(result)))
    return outputs, state_messages


# -------------------------
# Option B — explicit ReAct loop (manual tool execution)
# -------------------------
# replace your run loop with this robust streaming react loop
def run_agent_react_stream(user_query: str, session_state_messages: list):
    """
    Stream-based ReAct loop:
     - ask the LLM for next action with agent.stream(execute_tools=False)
     - if LLM returns a JSON tool-call or tool_calls metadata, execute the matching tool locally
     - feed result back as AIMessage and continue until final text produced
    """
    from time import time

    state_messages = list(session_state_messages) + [HumanMessage(content=user_query)]
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    outputs = []
    finished = False
    max_iterations = 8
    iteration = 0

    while not finished and iteration < max_iterations:
        iteration += 1
        # Use streaming but disable agent's automatic tool execution
        # Note: some versions accept execute_tools as part of config; others accept as kwarg.
        # Try both patterns safely.
        stream_kwargs = {"stream_mode": "values", "execute_tools": False}
        try:
            stream = agent.stream({"messages": state_messages}, config, **stream_kwargs)
        except TypeError:
            # fallback if agent.stream signature differs
            stream = agent.stream({"messages": state_messages}, config, stream_mode="values", execute_tools=False)

        lm_content = ""
        lm_msg_obj = None

        # collect latest message chunk from stream
        for event in stream:
            # each 'event' usually contains 'messages' list
            msgs = event.get("messages", []) if isinstance(event, dict) else getattr(event, "messages", [])
            if not msgs: 
                continue
            lm_msg_obj = msgs[-1]
            lm_content = getattr(lm_msg_obj, "content", "") or ""
            # break on the first non-empty message chunk for this iteration
            if lm_content:
                break

        if not lm_msg_obj and not lm_content:
            # nothing returned
            break

        # 1) Check for explicit JSON tool call inside text
        tool_call = None
        try:
            # attempt to extract JSON snippet
            tool_call = extract_json_from_text(lm_content)
        except Exception:
            tool_call = None

        # 2) If no JSON block, check for tool_calls metadata attached to message
        if tool_call is None:
            # many runtime message objects expose .tool_calls or .tool_calls metadata
            tc = None
            if hasattr(lm_msg_obj, "tool_calls"):
                tc = getattr(lm_msg_obj, "tool_calls")
            else:
                # sometimes runtime returns dict-like message inside content list
                tc = getattr(lm_msg_obj, "tool_calls", None) or (lm_msg_obj.extra.get("tool_calls") if getattr(lm_msg_obj, "extra", None) else None)
            # tc may be a list of tool call records; pick the first
            if tc:
                # standard shape: [{'name': 'air_quality_workflow', 'args': {...}, ... }]
                first = tc[0] if isinstance(tc, (list, tuple)) and tc else tc
                if isinstance(first, dict):
                    # unify to {"name":.., "parameters": ...}
                    if "args" in first and isinstance(first["args"], dict):
                        tool_call = {"name": first.get("name"), "parameters": first["args"]}
                    elif "args" in first and isinstance(first["args"], (list, tuple)):
                        tool_call = {"name": first.get("name"), "parameters": first["args"]}
                    elif "parameters" in first:
                        tool_call = {"name": first.get("name"), "parameters": first.get("parameters")}
                    else:
                        tool_call = None

        # If we found a tool_call, execute it locally
        if tool_call and isinstance(tool_call, dict) and "name" in tool_call:
            tname = tool_call["name"]
            params = tool_call.get("parameters", {})

            # find matching tool in your tool list (supports Tool-like objects or plain funcs)
            matching = None
            for t in tools:
                t_name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None) or (t.__name__ if callable(t) else None)
                if t_name == tname:
                    matching = t
                    break

            if not matching:
                err = {"error": f"tool '{tname}' not found locally"}
                outputs.append(err)
                state_messages.append(AIMessage(content=json.dumps(err)))
                continue

            # resolve callable
            tool_func = getattr(matching, "func", None) or (matching.get("func") if isinstance(matching, dict) else (matching if callable(matching) else None))
            if not callable(tool_func):
                err = {"error": f"tool '{tname}' has no callable func"}
                outputs.append(err)
                state_messages.append(AIMessage(content=json.dumps(err)))
                continue

            # execute — support dict params, positional list params, or single-string payload
            try:
                if isinstance(params, dict):
                    tool_result = tool_func(**params)
                elif isinstance(params, (list, tuple)):
                    # pass as positional args
                    tool_result = tool_func(*params)
                else:
                    # maybe a single JSON string
                    try:
                        parsed = json.loads(params)
                        if isinstance(parsed, dict):
                            tool_result = tool_func(**parsed)
                        elif isinstance(parsed, (list, tuple)):
                            tool_result = tool_func(*parsed)
                        else:
                            tool_result = tool_func(parsed)
                    except Exception:
                        tool_result = tool_func(params)
            except Exception as e:
                tool_result = {"error": f"exception while running tool '{tname}': {e}"}

            # append tool result for the user and feed back to the agent
            outputs.append(tool_result)
            # ensure text insertion is JSON-serializable
            try:
                tool_result_text = json.dumps(tool_result)
            except Exception:
                tool_result_text = str(tool_result)
            state_messages.append(AIMessage(content=tool_result_text))
            # continue loop so LLM can produce next action
            continue

        # No tool call -> treat LLM content as final textual answer
        outputs.append(lm_content)
        state_messages.append(AIMessage(content=lm_content))
        finished = True

    if iteration >= max_iterations:
        warn = {"warning": "max iterations reached"}
        outputs.append(warn)
        state_messages.append(AIMessage(content=json.dumps(warn)))

    return outputs, state_messages

def run_agent_debug_stream(user_query: str, session_state_messages: list):
    """
    Diagnostic: stream events from agent.stream and print their structure.
    Use this to inspect what the runtime actually returns (keys, types, nested fields).
    """
    state_messages = list(session_state_messages) + [HumanMessage(content=user_query)]
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Try a couple of common call signatures for agent.stream
    candidate_calls = [
        ({"messages": state_messages}, config, {"stream_mode": "values", "execute_tools": False}),
        ({"messages": state_messages}, config, {"stream_mode": "values"}),
        (state_messages, config, {"stream_mode": "values", "execute_tools": False}),
    ]

    for args, cfg, kwargs in candidate_calls:
        print("=== Trying agent.stream signature: args type:", type(args), "kwargs:", kwargs)
        try:
            stream = agent.stream(args, cfg, **kwargs)
        except TypeError as e:
            print("signature failed with TypeError:", e)
            continue
        except Exception as e:
            print("signature raised:", type(e), e)
            continue

        print("stream obtained, iterating events (will print up to 20)...")
        count = 0
        try:
            for event in stream:
                count += 1
                print(f"\n--- event #{count} ---")
                # print top-level event repr
                print("event repr:", repr(event)[:1000])
                # event may be dict-like or object; safely show keys/attrs
                if isinstance(event, dict):
                    print("event keys:", list(event.keys()))
                    # if event has messages, show each message's type and small repr
                    msgs = event.get("messages")
                else:
                    print("event attrs:", [a for a in dir(event) if not a.startswith("_")][:40])
                    msgs = getattr(event, "messages", None)

                print("messages container type:", type(msgs))
                if msgs:
                    # show last message details
                    last = msgs[-1]
                    print(" last msg repr (truncated):", repr(last)[:1000])
                    # print known helpful attrs if present
                    for attr in ("content", "tool_calls", "tool_call_id", "name", "extra"):
                        if hasattr(last, attr):
                            try:
                                val = getattr(last, attr)
                                print(f"  - {attr} type: {type(val)} -> repr: {repr(val)[:800]}")
                            except Exception as e:
                                print(f"  - {attr} error reading:", e)
                if count >= 20:
                    print("...reached 20 events, stopping")
                    break

        except Exception as e:
            print("exception iterating stream:", type(e), e)

        print("=== finished signature attempt ===\n\n")


# ---- robust react loop v2 ----
def run_agent_react_stream_v2(user_query: str, session_state_messages: list):
    """
    Robust streaming ReAct loop that:
      - tries several agent.stream signatures
      - prevents runtime from auto-executing tools (execute_tools=False when supported)
      - extracts JSON tool call from content or tool_calls metadata
      - executes local tool functions and feeds results back
      - avoids treating simple echo of user input as a final answer
    Returns (outputs_list, final_state_messages)
    """
    state_messages = list(session_state_messages) + [HumanMessage(content=user_query)]
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    outputs = []
    finished = False
    max_iterations = 8
    iteration = 0

    def get_stream():
        """Return a usable stream object by trying common call signatures."""
        sigs = [
            ({"messages": state_messages}, config, {"stream_mode": "values", "execute_tools": False}),
            ({"messages": state_messages}, config, {"stream_mode": "values"}),
            (state_messages, config, {"stream_mode": "values", "execute_tools": False}),
            (state_messages, config, {"stream_mode": "values"}),
        ]
        for args, cfg, kwargs in sigs:
            try:
                return agent.stream(args, cfg, **kwargs)
            except TypeError:
                # try next signature
                continue
            except Exception as e:
                # For unexpected errors, print and continue trying other signatures
                print("agent.stream attempt raised:", type(e), e)
                continue
        raise RuntimeError("Could not call agent.stream with tried signatures. Use run_agent_debug_stream to inspect available signatures.")

    while not finished and iteration < max_iterations:
        iteration += 1

        try:
            stream = get_stream()
        except Exception as e:
            # give helpful error to user and exit loop
            err = {"error": f"agent.stream unavailable: {e}"}
            outputs.append(err)
            state_messages.append(AIMessage(content=json.dumps(err)))
            break

        # collect a single model message for this iteration
        lm_msg_obj = None
        lm_content = ""

        try:
            for event in stream:
                # event can be dict-like or object-like; normalize
                msgs = None
                if isinstance(event, dict):
                    msgs = event.get("messages") or event.get("value") or event.get("data")
                else:
                    msgs = getattr(event, "messages", None) or getattr(event, "value", None)

                if not msgs:
                    # sometimes useful info lives directly on event
                    # print small debug hint for developers
                    # print("event (no messages):", type(event), list(event.keys()) if isinstance(event, dict) else None)
                    continue

                # last message chunk
                lm_msg_obj = msgs[-1]
                lm_content = getattr(lm_msg_obj, "content", "") or ""
                # Some runtimes first emit the human message back; ignore that if it's identical to last human
                last_human = state_messages[-1].content if state_messages and isinstance(state_messages[-1], HumanMessage) else ""
                if isinstance(lm_content, str) and lm_content.strip() == last_human.strip():
                    # ignore this echo and keep listening for the real response
                    # but if this is repeatedly the only thing we get, fallback after a short loop
                    # continue to next event
                    continue

                # break as soon as we have a non-echo content (even if it's JSON)
                break

        except Exception as e:
            # if stream iteration fails, surface error and exit
            err = {"error": f"error iterating stream: {e}"}
            outputs.append(err)
            state_messages.append(AIMessage(content=json.dumps(err)))
            break

        # if we didn't receive any meaningful lm_msg_obj/content
        if not lm_msg_obj and not lm_content:
            warn = {"warning": "No model message received from stream (empty). Try run_agent_debug_stream to inspect events."}
            outputs.append(warn)
            state_messages.append(AIMessage(content=json.dumps(warn)))
            break

        # Try to extract JSON tool-call from textual content
        tool_call = extract_json_from_text(lm_content)

        # If no JSON, check for tool_calls metadata on the message object
        if not tool_call:
            tc = getattr(lm_msg_obj, "tool_calls", None) or getattr(lm_msg_obj, "tool_call", None) or (getattr(lm_msg_obj, "extra", None) and getattr(lm_msg_obj, "extra").get("tool_calls") if getattr(lm_msg_obj, "extra", None) else None)
            if tc:
                first = tc[0] if isinstance(tc, (list, tuple)) and tc else tc
                if isinstance(first, dict):
                    if "args" in first and isinstance(first["args"], dict):
                        tool_call = {"name": first.get("name"), "parameters": first["args"]}
                    elif "parameters" in first:
                        tool_call = {"name": first.get("name"), "parameters": first.get("parameters")}
                    elif "args" in first and isinstance(first["args"], (list, tuple)):
                        tool_call = {"name": first.get("name"), "parameters": first["args"]}

        # If we have a tool_call -> execute locally
        if tool_call and isinstance(tool_call, dict) and "name" in tool_call:
            tname = tool_call["name"]
            params = tool_call.get("parameters", {})

            # find tool
            matching_tool = None
            for t in tools:
                t_name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None) or (t.__name__ if callable(t) else None)
                if t_name == tname:
                    matching_tool = t
                    break

            if not matching_tool:
                err = {"error": f"tool '{tname}' not found"}
                outputs.append(err)
                state_messages.append(AIMessage(content=json.dumps(err)))
                continue

            tool_func = getattr(matching_tool, "func", None) or (matching_tool if callable(matching_tool) else None) or (matching_tool.get("func") if isinstance(matching_tool, dict) else None)
            if not callable(tool_func):
                err = {"error": f"tool '{tname}' has no callable func"}
                outputs.append(err)
                state_messages.append(AIMessage(content=json.dumps(err)))
                continue

            try:
                if isinstance(params, dict):
                    tool_result = tool_func(**params)
                elif isinstance(params, (list, tuple)):
                    tool_result = tool_func(*params)
                else:
                    # try parsing if stringified
                    try:
                        parsed = json.loads(params)
                        if isinstance(parsed, dict):
                            tool_result = tool_func(**parsed)
                        elif isinstance(parsed, (list, tuple)):
                            tool_result = tool_func(*parsed)
                        else:
                            tool_result = tool_func(parsed)
                    except Exception:
                        tool_result = tool_func(params)
            except Exception as e:
                tool_result = {"error": f"exception in tool '{tname}': {e}"}

            outputs.append(tool_result)
            try:
                tool_text = json.dumps(tool_result)
            except Exception:
                tool_text = str(tool_result)
            state_messages.append(AIMessage(content=tool_text))
            # next iteration so model can act on tool_result
            continue

        # Otherwise, treat lm_content as final textual answer
        outputs.append(lm_content)
        state_messages.append(AIMessage(content=lm_content))
        finished = True

    if iteration >= max_iterations:
        warn = {"warning": "max iterations reached"}
        outputs.append(warn)
        state_messages.append(AIMessage(content=json.dumps(warn)))

    return outputs, state_messages

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example: assume no prior messages in session
    user_q = "Get me the air quality for Delhi, India for the next 3 days"
    session_msgs = []  # you may pass prior HumanMessage / AIMessage objects here

    # Try the manual react loop (Option B) to keep control of tool execution:
    outputs, final_state = run_agent_debug_stream(user_q, session_msgs)
    print("Outputs:", outputs)
