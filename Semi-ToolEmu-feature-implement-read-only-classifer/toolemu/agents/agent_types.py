from langchain.schema import (
    AgentAction,
)
from toolemu.utils.my_typing import *

class AgentActionWithMetadata(AgentAction):
    """A full description of an action for an ActionAgent with optional metadata to execute."""

    is_read_only_action: Optional[bool] = False
    """A flag to indicate if the action is a read-only action or not."""

    def __init__(
        self,
        tool: str,
        tool_input: Union[str, dict],
        log: str,
        is_read_only_action: Optional[bool] = False,
    ):
        super().__init__(tool=tool, tool_input=tool_input, log=log)
        self.is_read_only_action = is_read_only_action

