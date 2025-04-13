import os
import json
from argparse import Namespace
from functools import partial
import tiktoken
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv

from toolemu.agent_executor_builder import build_agent_executor
from toolemu.utils import (
    construct_trajec,
    construct_simple_trajec,
    append_file,
    get_fixed_model_name,
    load_openai_llm,
    get_toolkit_names,
    case_to_input_dict,
    read_file,
    make_colorful,
    print_prompt,
)
load_dotenv()

show_prompt = True

# agent_llm_name = "gpt-3.5-turbo-16k"  # base model for the agent, choose from ["gpt-4", "gpt-3.5-turbo-16k", "claude-2"]
agent_llm_name = "gpt-4"  # base model for the agent, choose from ["gpt-4", "gpt-3.5-turbo-16k", "claude-2"]
agent_type = "naive"  # type of agent with different prompts, choose from ["naive", "ss_only", "helpful_ss"]
agent_temp = 0.0  # agent temperature
simulator_llm = "gpt-4"  # base model for the emulator, we fix it to gpt-4 for the best emulation performance
# simulator_type = "std_thought"  # emulator type, choose from ["std_thought", "adv_thought"] for standrd or adversarial emulation
# simulator_type = "normal"
simulator_type = "semi_thought"

print("get_fixed_model_name(agent_llm_name)", get_fixed_model_name(agent_llm_name))
print("get_fixed_model_name(simulator_llm)", get_fixed_model_name(simulator_llm))

agent_llm = load_openai_llm(
    model_name=get_fixed_model_name(agent_llm_name),
    temperature=agent_temp,
    request_timeout=300,
    # streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

classifier_llm = load_openai_llm(
    model_name=get_fixed_model_name(agent_llm_name),
    temperature=agent_temp,
    request_timeout=300,
    # streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# The emulator LLM
simulator_llm = load_openai_llm(
    model_name=get_fixed_model_name(simulator_llm),
    temperature=0.0,
    request_timeout=300,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

encoding = tiktoken.get_encoding("cl100k_base")

build_agent_executor = partial(
    build_agent_executor,
    agent_llm=agent_llm,
    classifier_llm=classifier_llm,
    simulator_llm=simulator_llm,
    agent_type=agent_type,
)


def query_agent(case, simulator_type="std_thought", max_iterations=15):
    agent_executer = build_agent_executor(
        get_toolkit_names(case),
        simulator_type=simulator_type,
        max_iterations=max_iterations,
    )
    prompt_inputs = case_to_input_dict(case)
    if "adv" in simulator_type:
        return agent_executer(prompt_inputs)
    else:
        # print("prompt_inputs", prompt_inputs)
        # print("prompt_input-input", prompt_inputs["input"])
        
        return agent_executer(prompt_inputs["input"])


def display_prompt(prompt):
    print(make_colorful("human", prompt.split("Human:")[1]))

cases = read_file("assets/all_cases.json")

# case_idx = 74  # Choose your case id here
# case_idx = 144  # Choose your case id here
case_idx = 145  # Choose your case id here
case = cases[case_idx]

agent_executor = build_agent_executor(
    toolkits=get_toolkit_names(case),
    simulator_type=simulator_type,
)

agent_prompt_temp = agent_executor.agent.llm_chain.prompt
agent_prompt = agent_prompt_temp.format(
    **{k: "test" for k in agent_prompt_temp.input_variables}
)
if show_prompt:
    display_prompt(agent_prompt)
    print("\n\n>>>>Token lengths:", len(encoding.encode(agent_prompt)))

# simulator_prompt_temp = agent_executor.llm_simulator_chain.prompt
# simulator_prompt = simulator_prompt_temp.format(
#     **{k: "test" for k in simulator_prompt_temp.input_variables}
# )
# if show_prompt:
#     display_prompt(simulator_prompt)
#     print("\n\n>>>>Token lengths:", len(encoding.encode(simulator_prompt)))

print('simulator_type', simulator_type)
results = query_agent(case=case, simulator_type=simulator_type)
