from .tool_interface import BaseToolkit

__REAL_TOOLKITS_DICT__ = {}
__VIRTUAL_TOOLKITS_DICT__ = {}

__TOOL_TYPE_DICT__ = {
    "REAL": "Real",
    "VIRTUAL": "Virtual",
}
tool_types = list(__TOOL_TYPE_DICT__.values())

__TOOLKITS_DICT__ = {
    __TOOL_TYPE_DICT__["REAL"]: __REAL_TOOLKITS_DICT__,
    __TOOL_TYPE_DICT__["VIRTUAL"]: __VIRTUAL_TOOLKITS_DICT__,
}


def get_toolkit_dict(type):
    return __TOOLKITS_DICT__.get(type, {})


def toolkits_factory(name, type):
    return __TOOLKITS_DICT__.get(type, {}).get(name, None)


def register_toolkit(type, overwrite=None):
    def register_function_fn(cls):
        if type not in tool_types:
            raise ValueError(f"Tool Type {type} is not recognized!")

        name = overwrite
        toolkit_dict = __TOOLKITS_DICT__[type]
        if name is None:
            name = cls.__name__
        if name in toolkit_dict:
            raise ValueError(f"The {type} tool - Name {name} is already registered!")
        if not issubclass(cls, BaseToolkit):
            raise ValueError(f"Class {cls} is not a subclass of {BaseToolkit}")
        toolkit_dict[name] = cls
        print(f"A {type} Toolkit registered: [{name}]")
        return cls

    return register_function_fn
