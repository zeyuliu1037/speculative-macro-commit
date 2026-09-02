"""Baseline AppWorld agent using tool-calling format via vLLM.

Uses a generic execute_api tool to call any AppWorld API,
plus discovery tools (show_api_descriptions, show_api_doc, etc.)
and supervisor shortcuts.
"""

import ctypes
import ctypes.util
import json
import traceback
from typing import Any, Optional

# AppWorld uses freezegun which freezes time.time() AND time.perf_counter().
# Use ctypes clock_gettime to bypass the freeze.
_CLOCK_MONOTONIC = 1
class _Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]
_librt = ctypes.CDLL(ctypes.util.find_library('rt') or 'librt.so.1')
_ts = _Timespec()

def perf_counter():
    _librt.clock_gettime(_CLOCK_MONOTONIC, ctypes.byref(_ts))
    return _ts.tv_sec + _ts.tv_nsec / 1e9

from openai import OpenAI

from .tools import TOOLS
from .trajectory_logger import TrajectoryLogger


DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

SYSTEM_PROMPT = """You are a helpful assistant that completes tasks by calling APIs on various apps.

You have access to the following tools:
- show_app_descriptions: List all available apps
- show_api_descriptions: List APIs for a specific app
- show_api_doc: Get full documentation for a specific API
- search_api_docs: Search for APIs by description
- execute_api: Call any API with parameters
- show_account_passwords: Get login credentials
- show_active_task: See the current task
- complete_task: Submit your final answer
- show_profile / show_addresses / show_payment_cards: Get personal info

Workflow:
1. Read the task carefully
2. Use show_account_passwords to get credentials if needed
3. Login to relevant apps using execute_api
4. Use show_api_descriptions and show_api_doc to discover available APIs
5. Call APIs to gather information or perform actions
6. When done, call complete_task with the answer

Important:
- Always login to an app before using its APIs
- Use show_api_doc to check required parameters before calling execute_api
- Be precise with parameter names and types
"""


def _get_client(api_base: str) -> OpenAI:
    return OpenAI(base_url=api_base, api_key="dummy", timeout=300.0, max_retries=2)


def _thinking_body(thinking: bool, thinking_budget: int = 4096):
    if not thinking:
        return DISABLE_THINKING
    body = {"chat_template_kwargs": {"enable_thinking": True}}
    if thinking_budget > 0:
        body["chat_template_kwargs"]["thinking_budget"] = thinking_budget
    return body


def _parse_tool_call(message) -> tuple[str, dict]:
    """Extract tool name and arguments from response message."""
    if message.tool_calls and len(message.tool_calls) > 0:
        tc = message.tool_calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
        except (json.JSONDecodeError, TypeError):
            args = {}
        return name, args
    return "", {}


def _parse_qwen_tool_call(content: str) -> tuple[str, dict]:
    """Parse Qwen3.5's XML-style tool calls from content text."""
    import re
    if not content or "<tool_call>" not in content:
        return "", {}
    match = re.search(
        r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>',
        content, re.DOTALL
    )
    if not match:
        return "", {}
    func_name = match.group(1)
    params_text = match.group(2)
    params = {}
    for param_match in re.finditer(
        r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', params_text, re.DOTALL
    ):
        key = param_match.group(1)
        value = param_match.group(2).strip()
        try:
            value = json.loads(value)
        except (ValueError, json.JSONDecodeError):
            pass
        params[key] = value
    return func_name, params


def execute_tool(env, tool_name: str, tool_args: dict) -> str:
    """Execute a tool call on the AppWorld environment.

    Returns the string result to send back to the LLM.
    """
    try:
        if tool_name == "show_app_descriptions":
            result = env.execute("import json; print(json.dumps(apis.api_docs.show_app_descriptions(), indent=2))")
        elif tool_name == "show_api_descriptions":
            app = tool_args.get("app_name", "")
            result = env.execute(f'import json; print(json.dumps(apis.api_docs.show_api_descriptions(app_name="{app}"), indent=2))')
        elif tool_name == "show_api_doc":
            app = tool_args.get("app_name", "")
            api = tool_args.get("api_name", "")
            result = env.execute(f'import json; print(json.dumps(apis.api_docs.show_api_doc(app_name="{app}", api_name="{api}"), indent=2))')
        elif tool_name == "search_api_docs":
            query = tool_args.get("query", "")
            result = env.execute(f'import json; print(json.dumps(apis.api_docs.search_api_docs(query="{query}"), indent=2))')
        elif tool_name == "show_account_passwords":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_account_passwords(), indent=2))")
        elif tool_name == "show_active_task":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_active_task(), indent=2))")
        elif tool_name == "show_profile":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_profile(), indent=2))")
        elif tool_name == "show_addresses":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_addresses(), indent=2))")
        elif tool_name == "show_payment_cards":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_payment_cards(), indent=2))")
        elif tool_name == "complete_task":
            output = tool_args.get("output", "")
            # Escape for Python string
            output_escaped = output.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            result = env.execute(f'import json; print(json.dumps(apis.supervisor.complete_task(output="{output_escaped}"), indent=2))')
        elif tool_name == "execute_api":
            app = tool_args.get("app_name", "")
            api = tool_args.get("api_name", "")
            params = tool_args.get("parameters", {})
            # Build kwargs string
            kwargs_parts = []
            for k, v in params.items():
                if isinstance(v, str):
                    v_escaped = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                    kwargs_parts.append(f'{k}="{v_escaped}"')
                elif isinstance(v, bool):
                    kwargs_parts.append(f'{k}={v}')
                elif v is None:
                    kwargs_parts.append(f'{k}=None')
                else:
                    kwargs_parts.append(f'{k}={json.dumps(v)}')
            kwargs_str = ", ".join(kwargs_parts)
            code = f'import json; print(json.dumps(apis.{app}.{api}({kwargs_str}), indent=2))'
            result = env.execute(code)
        else:
            result = f"Unknown tool: {tool_name}"
    except Exception as e:
        result = f"Error executing {tool_name}: {e}"
    return result


def _build_canonical_message(step, tool_name, tool_args, tool_result):
    """Build canonical (assistant, tool) message pair for committed history.

    Single source of truth for the message shape so baseline and pipeline
    grow `messages` bit-identically — divergence here causes greedy decoding
    to walk different trajectories on subsequent steps.

    sort_keys=True normalizes JSON key order so that semantically equal args
    from different sources (owner 27B vs speculator 4B) produce byte-identical
    strings. Without this, peer spec_ctx (built from s1's key order) can
    diverge from committed messages (built from owner's key order).
    """
    tc_id = f"call_{step}"
    canon_args = json.dumps(tool_args, sort_keys=True)
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": canon_args},
        }],
    }
    tool_msg = {
        "role": "tool",
        "tool_call_id": tc_id,
        "name": tool_name,
        "content": tool_result[:4000] if isinstance(tool_result, str) else str(tool_result)[:4000],
    }
    return assistant_msg, tool_msg


def run_agent(
    env,
    task_instruction: str,
    model: str,
    api_base: str,
    thinking: bool = False,
    thinking_budget: int = 4096,
    max_steps: int = 20,
    temperature: float = 0.0,
    logger: Optional[TrajectoryLogger] = None,
    tracer=None,
) -> dict:
    """Run the baseline agent on one AppWorld task.

    Returns dict with reward, n_steps, messages, timing info.
    """
    from .event_trace import NULL_TRACER, messages_fingerprint
    tracer = tracer if tracer is not None else NULL_TRACER
    client = _get_client(api_base)
    extra_body = _thinking_body(thinking, thinking_budget)
    max_tokens = (thinking_budget + 4096) if thinking else 1024

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task_instruction}"},
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    step_times = []
    task_completed = False

    for step in range(1, max_steps + 1):
        t_start = perf_counter()
        schema_error = False
        execution_error = False
        thinking_truncated = False
        tool_name = ""
        tool_args = {}
        tool_result = ""

        try:
            tracer.emit("owner", "llm_submit",
                        step_idx=step,
                        prefix_len=len(messages),
                        prefix_hash=messages_fingerprint(messages))
            with tracer.span("owner", "llm_call", step_idx=step):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    temperature=temperature,
                    extra_body=extra_body,
                    max_tokens=max_tokens,
                )
            tracer.emit("owner", "llm_end", step_idx=step)
            msg = response.choices[0].message
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            # Parse tool call
            tool_name, tool_args = _parse_tool_call(msg)

            # Fallback: parse Qwen XML format from content
            if not tool_name and msg.content:
                tool_name, tool_args = _parse_qwen_tool_call(msg.content)
                if tool_name:
                    # Check if thinking consumed all tokens
                    if not tool_name and "<think>" in (msg.content or ""):
                        thinking_truncated = True

            if not tool_name:
                # No tool call — model may be responding with text
                content = msg.content or ""
                if not content.strip():
                    thinking_truncated = True
                    schema_error = True
                    tool_name = "_no_tool_call"
                    tool_result = "Error: No tool call produced. Please call a tool."
                else:
                    # Model produced text without tool call — treat as respond
                    tool_name = "_text_response"
                    tool_result = content
                    # Add as assistant message and continue
                    messages.append({"role": "assistant", "content": content})
                    t_end = perf_counter()
                    if logger:
                        logger.log_step(
                            step_id=step,
                            instruction=task_instruction[:500],
                            model_output=content[:2000],
                            api_name=tool_name,
                            api_params={},
                            api_result=tool_result[:2000],
                            wall_clock_time=t_end - t_start,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            schema_error=schema_error,
                            thinking_truncated=thinking_truncated,
                        )
                    continue

            # Execute tool
            with tracer.span("env", "env_exec", who="main", tool=tool_name):
                tool_result = execute_tool(env, tool_name, tool_args)
            if "Error" in tool_result and "Traceback" in tool_result:
                execution_error = True
            tracer.emit("main", "commit", step_idx=step, tool=tool_name)

            assistant_msg, tool_response_msg = _build_canonical_message(
                step, tool_name, tool_args, tool_result)
            messages.append(assistant_msg)
            messages.append(tool_response_msg)

            # Check if task completed
            if tool_name == "complete_task":
                task_completed = True

        except Exception as e:
            tool_name = "_error"
            tool_result = f"Exception: {traceback.format_exc()}"
            execution_error = True
            input_tokens = 0
            output_tokens = 0

        t_end = perf_counter()
        step_time = t_end - t_start
        step_times.append(step_time)

        if logger:
            logger.log_step(
                step_id=step,
                instruction=task_instruction[:500],
                model_output=json.dumps({"tool": tool_name, "args": tool_args})[:2000],
                api_name=tool_name,
                api_params=tool_args,
                api_result=tool_result[:2000],
                wall_clock_time=step_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                schema_error=schema_error,
                execution_error=execution_error,
                thinking_truncated=thinking_truncated,
            )

        print(f"  Step {step}: {tool_name}({json.dumps(tool_args)[:80]}) -> {tool_result[:100]}... [{step_time:.2f}s]")

        if task_completed:
            break

    # Evaluate using AppWorld's TestTracker
    try:
        eval_result = env.evaluate(suppress_errors=True)
        reward = 1.0 if eval_result.success else 0.0
        pass_rate = eval_result.pass_percentage / 100.0 if eval_result.total_count > 0 else 0.0
        print(f"  Eval: success={eval_result.success} "
              f"passed={eval_result.pass_count}/{eval_result.total_count} "
              f"completed={eval_result.task_completed}")
        # Use pass_rate as the score for partial credit
        reward = pass_rate
    except Exception as e:
        print(f"  Evaluation error: {e}")
        reward = 0.0

    if logger:
        logger.finalize(task_success=(reward > 0), evaluation_score=reward)

    return {
        "reward": reward,
        "n_steps": len(step_times),
        "task_completed": task_completed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_step_time": sum(step_times) / len(step_times) if step_times else 0,
        "total_wall_time": sum(step_times),
        "messages": messages,
    }
