from .real_tools import *
from .register import get_toolkit_dict, toolkits_factory, __TOOL_TYPE_DICT__
from .tool_interface import BaseToolkit, FunctionToolkit
from .virtual_tools import *


def get_toolkits_by_names_n_type(names: List[str], type: str = __TOOL_TYPE_DICT__["VIRTUAL"]) -> List[FunctionToolkit]:
    toolkits = []
    for name in names:
        # Try to get the toolkit by name, if not found, try to get it by name with type prefix
        toolkit = toolkits_factory(name, type) or toolkits_factory(f'{type}{name}', type)
        if toolkit:
            toolkits.append(toolkit())
        else:
            print(f"Warning: toolkit {name} not found")
    return toolkits

def get_virtual_toolkits_by_names(names: List[str]) -> List[FunctionToolkit]:
    return get_toolkits_by_names_n_type(names, __TOOL_TYPE_DICT__["VIRTUAL"])

def get_real_toolkits_by_names(names: List[str]) -> List[FunctionToolkit]:
    return get_toolkits_by_names_n_type(names, __TOOL_TYPE_DICT__["REAL"])

def get_toolkits_by_names(names: List[str]) -> List[FunctionToolkit]:
    return get_virtual_toolkits_by_names(names) + get_real_toolkits_by_names(names)

def get_tool_class_by_name(toolkits: List[FunctionToolkit], name: str) -> BaseTool:
    for toolkit in toolkits:
        try:
            return toolkit[name]
        except ValueError:
            pass
    raise ValueError(f"Tool {name} does not exist in these toolkits")
