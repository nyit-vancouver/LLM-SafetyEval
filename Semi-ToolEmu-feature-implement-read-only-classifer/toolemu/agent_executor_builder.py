from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models.base import BaseChatModel
from toolemu.agents import (
    __SIMULATOR_TYPES_DICT__ as SIMULATOR_TYPES,
    SIMULATORS,
    AgentExecutorWithToolkit,
    ZeroShotAgentWithToolkit,
)
from toolemu.tools import (get_virtual_toolkits_by_names, get_real_toolkits_by_names)
from toolemu.utils.my_typing import *


def build_agent_executor(
    toolkits: List[str],
    agent_llm: BaseLanguageModel,
    classifier_llm: BaseLanguageModel,
    simulator_llm: BaseLanguageModel,
    critiquer_llm: Optional[BaseLanguageModel] = None,
    num_critique_steps: int = 0,
    max_allowed_steps: int = 3,
    agent_type: str = "naive",
    simulator_type: str = "std_thought",
    verbose: bool = True,
    return_intermediate_steps: bool = True,
    max_iterations: int = 15,
    callback_manager: Optional[BaseCallbackManager] = None,
) -> Type[AgentExecutorWithToolkit]:
    # Need to both get the real and virtual toolkits separately here to identify which agent should be used
    # Add a new {simulator_type} to support semi-executor
    virtual_toolkits = get_virtual_toolkits_by_names(toolkits)
    real_toolkits = get_real_toolkits_by_names(toolkits)
    
    print("======Applied Virtual Toolkits======")
    print(virtual_toolkits)
    print("======Applied Real Toolkits======")
    print(real_toolkits)

    agent = ZeroShotAgentWithToolkit.from_llm_and_toolkits(
        toolkits=virtual_toolkits + real_toolkits,
        llm=agent_llm,
        classifier_llm=classifier_llm,
        agent_type=agent_type,
        use_chat_format=isinstance(agent_llm, BaseChatModel),
    )

    # if simulator_type not in SIMULATOR_TYPES.values():

    if simulator_type == SIMULATOR_TYPES["NORMAL"]:
        return SIMULATORS[SIMULATOR_TYPES["NORMAL"]].from_agent_and_toolkits(
            toolkits=real_toolkits,
            agent=agent,
            verbose=verbose,
            max_iterations=max_iterations,
            callback_manager=callback_manager,
        )
    
    if simulator_type == SIMULATOR_TYPES["STD_THOUGHT"]:
        return SIMULATORS[SIMULATOR_TYPES["STD_THOUGHT"]].from_agent_and_toolkits(
            toolkits=virtual_toolkits,
            agent=agent,
            verbose=verbose,
            llm_simulator=simulator_llm,
            llm_critiquer=critiquer_llm,
            num_critique_steps=num_critique_steps,
            max_allowed_steps=max_allowed_steps,
            use_chat_format=isinstance(simulator_llm, BaseChatModel),
            return_intermediate_steps=return_intermediate_steps,
            max_iterations=max_iterations,
            callback_manager=callback_manager,
        )
    
    if simulator_type == SIMULATOR_TYPES["ADV_THOUGHT"]:
        return SIMULATORS[SIMULATOR_TYPES["ADV_THOUGHT"]].from_agent_and_toolkits(
            toolkits=virtual_toolkits,
            agent=agent,
            verbose=verbose,
            llm_simulator=simulator_llm,
            llm_critiquer=critiquer_llm,
            num_critique_steps=num_critique_steps,
            max_allowed_steps=max_allowed_steps,
            use_chat_format=isinstance(simulator_llm, BaseChatModel),
            return_intermediate_steps=return_intermediate_steps,
            max_iterations=max_iterations,
            callback_manager=callback_manager,
        )
    
    if simulator_type == SIMULATOR_TYPES["SEMI_THOUGHT"]:
        return SIMULATORS[SIMULATOR_TYPES["SEMI_THOUGHT"]].from_agent_and_toolkits(
            # virtual_toolkits=virtual_toolkits,
            # real_toolkits=real_toolkits,
            toolkits=virtual_toolkits + real_toolkits,
            agent=agent,
            verbose=verbose,
            llm_simulator=simulator_llm,
            llm_critiquer=critiquer_llm,
            num_critique_steps=num_critique_steps,
            max_allowed_steps=max_allowed_steps,
            use_chat_format=isinstance(simulator_llm, BaseChatModel),
            return_intermediate_steps=return_intermediate_steps,
            max_iterations=max_iterations,
            callback_manager=callback_manager,
        )
    
