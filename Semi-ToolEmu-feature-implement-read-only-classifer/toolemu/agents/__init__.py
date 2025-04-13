from .agent_executor import AgentExecutorWithToolkit
from .virtual_agent_executor import (
    AdversarialVirtualAgentExecutorWithToolkit,
    StandardVirtualAgentExecutorWithToolkit,
)
from .semi_executor import SemiAgentExecutorWithToolkit
from .zero_shot_agent_with_toolkit import AGENT_TYPES, ZeroShotAgentWithToolkit
from .agent_types import AgentActionWithMetadata


__SIMULATOR_TYPES_DICT__ = {
    "NORMAL": "normal",
    "STD_THOUGHT": "std_thought",
    "ADV_THOUGHT": "adv_thought",
    "SEMI_THOUGHT": "semi_thought",
}

SIMULATORS = {
    __SIMULATOR_TYPES_DICT__["NORMAL"]: AgentExecutorWithToolkit,
    __SIMULATOR_TYPES_DICT__["STD_THOUGHT"]: StandardVirtualAgentExecutorWithToolkit,
    __SIMULATOR_TYPES_DICT__["ADV_THOUGHT"]: AdversarialVirtualAgentExecutorWithToolkit,
    __SIMULATOR_TYPES_DICT__["SEMI_THOUGHT"]: SemiAgentExecutorWithToolkit
}
