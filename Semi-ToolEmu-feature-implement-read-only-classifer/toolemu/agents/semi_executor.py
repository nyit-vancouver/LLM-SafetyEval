import json
import re
import time

from langchain.agents import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    Tool,
    tool,
)
from langchain.agents.agent import ExceptionTool
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    BaseRunManager,
    CallbackManager,
    CallbackManagerForChainRun,
)
from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.input import get_color_mapping
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseStringMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseMessage,
    BaseOutputParser,
    HumanMessage,
    SystemMessage,
    OutputParserException
)
from langchain.tools.base import BaseTool, StructuredTool
from langchain.utilities.asyncio import asyncio_timeout
from procoder.functional import collect_refnames, format_multiple_prompts, format_prompt
from procoder.prompt import Module as PromptModule
from pydantic import BaseModel
from toolemu.prompts.simulator import (
    ADV_SIMULATOR_CRITIQUE,
    ADV_SIMULATOR_CRITIQUE_REPEAT,
    ADV_SIMULATOR_PROMPT,
    ADV_SIMULATOR_SYSTEM_INFO,
    STD_SIMULATOR_CRITIQUE,
    STD_SIMULATOR_CRITIQUE_REPEAT,
    STD_SIMULATOR_PROMPT,
    STD_SIMULATOR_SYSTEM_INFO,
)
from toolemu.tools import RealHuman, RealHumanAssistanceQuery
from toolemu.tools.tool_interface import BaseToolkit
from toolemu.utils import InvalidTool, run_with_input_validation, validate_outputs
from toolemu.utils.my_typing import *

from .agent_executor import AgentExecutorWithToolkit
from .agent_types import AgentActionWithMetadata

Prompt = Union[PromptModule, str]

class SimulatorInputModel(BaseModel):
    simulator_scratchpad: Optional[Any]
    current_tool: Optional[str]
    current_tool_description: Optional[str]
    toolkit_descriptions: Optional[str]
    input: Optional[str]
    underspecifications: Optional[str]
    risky_outcome: Optional[str]
    risky_actions: Optional[str]


# TODO: decouple the simulator from the agent executor

class SemiAgentExecutorWithToolkit(AgentExecutorWithToolkit):
    # """Virtual agent executor that simulates the execution of virtual tools with LLMs"""
    """Virtual agent executor that outputs thoughts before simulating the execution of virtual tools."""

    llm_simulator_chain: LLMChain
    llm_critiquer: Optional[BaseLanguageModel] = None
    num_critique_steps: Optional[int] = 0
    max_allowed_steps: Optional[int] = 3
    sim_system_info: Prompt = STD_SIMULATOR_SYSTEM_INFO
    sim_prompt_instruction: Prompt = STD_SIMULATOR_PROMPT
    critique_prompt: Prompt = STD_SIMULATOR_CRITIQUE
    critique_prompt_repeat: Prompt = STD_SIMULATOR_CRITIQUE_REPEAT
    _input_keys: List[str] = ["input"]

    @classmethod
    def from_agent_and_toolkits(
        cls,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        # virtual_toolkits: Sequence[BaseToolkit],
        # real_toolkits: Sequence[BaseToolkit],
        toolkits: Sequence[BaseToolkit],
        llm_simulator: BaseLanguageModel,
        llm_critiquer: Optional[BaseLanguageModel] = None,
        num_critique_steps: Optional[int] = 0,
        max_allowed_steps: Optional[int] = 3,
        callback_manager: Optional[BaseCallbackManager] = None,
        use_chat_format: Optional[bool] = False,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create from agent and toolkits."""
        tools = agent.get_all_tools(toolkits)
        tool_names = [tool.name for tool in tools]
        if use_chat_format:
            assert isinstance(llm_simulator, BaseChatModel)

        simulator_prompt = cls.create_simulator_prompt(use_chat_format=use_chat_format)
        llm_simulator_chain = LLMChain(
            llm=llm_simulator,
            prompt=simulator_prompt,
            callback_manager=callback_manager,
        )

        # NOTE: change to use the simulator as the default critiquer
        if llm_critiquer is None:
            llm_critiquer = llm_simulator

        return cls(
            agent=agent,
            tools=tools,
            toolkits=toolkits,
            tool_names=tool_names,
            llm_simulator_chain=llm_simulator_chain,
            llm_critiquer=llm_critiquer,
            num_critique_steps=num_critique_steps,
            max_allowed_steps=max_allowed_steps,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def get_var(cls, name):
        """Get the default value of a class variable of Pydantic model."""
        return cls.__fields__[name].default

    @classmethod
    def create_simulator_prompt(
        cls, use_chat_format: Optional[bool] = False
    ) -> BasePromptTemplate:
        """Create a the prompt for the simulator LLM."""
        inputs = dict()
        system_info = cls.get_var("sim_system_info")
        prompt_instruction = cls.get_var("sim_prompt_instruction")
        system_info, prompt_instruction = format_multiple_prompts(
            [system_info, prompt_instruction], inputs, include_brackets=[False, True]
        )

        if use_chat_format:
            simulator_system_message = SystemMessage(content=system_info)
            simulator_instruction_message = HumanMessagePromptTemplate.from_template(
                template=prompt_instruction
            )

            messages = [
                simulator_system_message,
                simulator_instruction_message,
            ]
            return ChatPromptTemplate.from_messages(messages=messages)
        else:
            template = "\n\n".join([system_info, prompt_instruction])

            input_variables = cls.get_var("_input_keys") + ["simulator_scratchpad"]
            return PromptTemplate(template=template, input_variables=input_variables)

    def _get_current_toolkit_descriptions(self, tool_name: str) -> str:
        # NOTE: assume only one toolkit has the tool with tool_name
        for toolkit in self.toolkits:
            for tool in toolkit.tools:
                if tool.name == tool_name:
                    return toolkit.create_description(detail_level="low")
        raise ValueError(f"Tool {tool_name} not found in any of the toolkits.")

    @property
    def input_keys(self) -> List[str]:
        return self._input_keys

    @property
    def generatetion_prefix(self) -> str:
        return "Simulator Thought: "

    @property
    def thought_summary_prefix(self) -> str:
        return "Simulator Log Summary: "

    @property
    def stop_seqs(self) -> List[str]:
        return [
            "\nThought:",
            "\n\tThought:",  # or {agent.llm_prefix.rstrip()}
            "\nAction:",
            "\n\tAction:",
        ]

    @property
    def llm_simulator_tool(self) -> BaseTool:
        result = StructuredTool.from_function(
            func=lambda callbacks, **kwargs: self._get_simulated_observation(
                callbacks, **kwargs
            ),
            name="llm_simulator",
            description="Simulate the execution of a tool with a language model",
            args_schema=SimulatorInputModel
            # infer_schema=False
        )
        return result

    def _fix_observation_text(self, text: str):
        return text.rstrip() + "\n"

    def _extract_observation_and_thought(self, llm_output: str) -> Optional[List[str]]:
        """Parse out the observation from the LLM output."""
        # \s matches against tab/newline/whitespace
        regex = rf"{self.thought_summary_prefix}(.*?)[\n]*{self.agent.observation_prefix}[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if not match:
            return None
        thought_summary = match.group(1).strip()
        observation = match.group(2).strip()
        return observation, thought_summary

    def _get_simulated_observation(
        self, callback_manager: CallbackManager, **full_inputs: Any
    ) -> SimulatedObservation:
        streaming_output = self.llm_simulator_chain.llm.streaming
        if streaming_output:
            print("\n" + self.generatetion_prefix)
            # for handler in callback_manager.handlers:
            #     getattr(handler, "on_text")(
            #         "\n" + self.generatetion_prefix, verbose=self.verbose
            #     )

        # Execute the emulated tool here
        full_output = self.llm_simulator_chain.predict(
            **full_inputs, stop=self.stop_seqs
        )
        parsed_output = self._extract_observation_and_thought(full_output)
        while parsed_output is None:
            full_output = self._fix_observation_text(full_output)
            full_inputs["simulator_scratchpad"] += full_output
            output = self.llm_simulator_chain.predict(
                **full_inputs, stop=self.stop_seqs
            )
            full_output += output
            parsed_output = self._extract_observation_and_thought(full_output)

        log_output = self.generatetion_prefix + full_output
        # remove all the text after self.agent.observation_prefix
        log_output = log_output.split(self.agent.observation_prefix)[0].strip()
        log_output = "\n" + log_output

        if not streaming_output and not log_output.isspace():
            for handler in callback_manager.handlers:
                getattr(handler, "on_tool_end")(log_output, verbose=self.verbose)

        # Here 
        sim_observation = SimulatedObservation(
            observation=parsed_output[0],
            thought_summary=parsed_output[1],
            log=full_output,
        )

        observation = self._critique_simulated_observation(
            callback_manager, sim_observation, full_inputs
        )
        return observation

    def _construct_simulator_scratchpad(
        self,
        is_read_only_action: bool,
        intermediate_steps: List[Tuple[AgentAction, str]],
        include_simulator_log: bool = False,
        include_simulator_thought_summary: bool = True,
        include_simulator_last_step_only: bool = False,
    ):
        """Construct the scratchpad that without outputting the last observation."""

        # this is copied from the agent's _construct_scratchpad
        scratchpad = ""
        for idx, (action, observation) in enumerate(intermediate_steps):
            scratchpad += f"Action: {action.tool}\nAction Input: {action.tool_input}\n"

            if idx == len(intermediate_steps) - 1 or not is_read_only_action:
                scratchpad += "\n"
            else:
                if include_simulator_log and (
                    not include_simulator_last_step_only
                    or idx == len(intermediate_steps) - 2
                ):
                    scratchpad += f"\n{self.generatetion_prefix}{observation.log}\n"
                elif include_simulator_thought_summary and (
                    not include_simulator_last_step_only
                    or idx == len(intermediate_steps) - 2
                ):
                    scratchpad += f"\n{self.thought_summary_prefix}{observation.thought_summary}\n{self.agent.observation_prefix}{observation.observation}\n"
                else:
                    scratchpad += (
                        f"\n{self.agent.observation_prefix}{observation.observation}\n"
                    )
                # scratchpad += self.agent.llm_prefix

        # add prefix for generation
        scratchpad += self.generatetion_prefix
        # scratchpad = self.agent.llm_prefix + scratchpad

        return scratchpad

    def _create_critiquer_prompt(
        self,
        simulator_inputs: Dict[str, str],
        sim_observation: SimulatedObservation,
        critique_outputs: List[Dict[str, str]],
    ) -> BasePromptTemplate:
        """Create a the prompt for the critiquer LLM."""
        refnames = collect_refnames(
            dict(
                sim_prompt=self.sim_prompt_instruction, crit_prompt=self.critique_prompt
            ),
        )
        critique_prompt = format_prompt(
            self.critique_prompt, {}, refnames=refnames, include_brackets=True
        )
        critique_prompt_repeat = format_prompt(
            self.critique_prompt_repeat, {}, refnames=refnames, include_brackets=True
        )

        simulator_prompt_temp = self.llm_simulator_chain.prompt
        use_chat_format = isinstance(simulator_prompt_temp, ChatPromptTemplate)
        simulator_prompt = simulator_prompt_temp.format_prompt(**simulator_inputs)

        critique_prompt_messages = []

        if use_chat_format:
            # add simulator prompt
            critique_prompt_messages += simulator_prompt.messages
        else:
            # add simulator prompt
            critique_prompt_messages.append(HumanMessage(content=simulator_prompt))

        # add simulator output
        simulator_output = sim_observation.log
        critique_prompt_messages.append(AIMessage(content=simulator_output))

        # The last dict in critique_outputs only contains the validation results
        for idx, crit_dict in enumerate(critique_outputs):
            prompt = critique_prompt if idx == 0 else critique_prompt_repeat
            prompt = f"{crit_dict['validation']}\n{prompt}"
            critique_prompt_messages.append(HumanMessage(content=prompt))
            if "critique" in crit_dict:
                # add critique output
                critique_prompt_messages.append(
                    AIMessage(content=crit_dict["critique"])
                )

        if not use_chat_format:
            critique_prompt_messages = "\n\n".join(
                [t.content for t in critique_prompt_messages]
            )

        return critique_prompt_messages

    @property
    def critique_prefix(self) -> str:
        return "Critique #{step}:"

    @property
    def revised_thought_summary_prefix(self) -> str:
        return "Revised Simulator Log Summary #{step}:"

    @property
    def revised_observation_prefix(self) -> str:
        return "Revised Observation #{step}:"

    def _extract_revised_observation_and_thought(
        self, critique_llm_output: str, current_step: int
    ) -> Optional[List[str]]:
        """Parse out the observation from the critiqued LLM output."""
        thought_summary_prefix = self.revised_thought_summary_prefix.format(
            step=current_step
        )
        observation_prefix = self.revised_observation_prefix.format(step=current_step)
        # \s matches against tab/newline/whitespace
        regex = rf"{thought_summary_prefix}(.*?)[\n]*{observation_prefix}[\s]*(.*)"
        match = re.search(regex, critique_llm_output, re.DOTALL)

        if not match:
            return None
        revised_thought_summary = match.group(1).strip()
        revised_observation = match.group(2).strip()
        return revised_observation, revised_thought_summary

    def _critique_simulated_observation(
        self,
        callback_manager: CallbackManager,
        sim_observation: SimulatedObservation,
        simulator_inputs: Dict[str, Any],
    ):
        streaming_output = self.llm_critiquer.streaming

        tool_name = simulator_inputs["current_tool"]
        tool_mapping = dict(zip(self.tool_names, self.tools))
        tool = tool_mapping[tool_name]

        def get_validation_result(obs):
            msg = "The format of the output matches the specification of the tool."
            exception = None
            try:
                outputs = json.loads(obs)
            except json.decoder.JSONDecodeError as e:
                msg = f"The output is not a valid JSON object."
                exception = e
            if exception is None:
                try:
                    validate_outputs(tool.returns, outputs)
                except ValueError as e:
                    msg = f"The format of the output does not match the specification of the tool."
                    exception = e
            return f"Format Validation: {msg}", exception

        current_obs = sim_observation.observation
        critique_outputs = []
        sep = "\n\n"
        revised_output = None

        if self.max_allowed_steps <= 0:
            return sim_observation

        for step in range(self.max_allowed_steps):
            step_idx = step + 1

            validation_msg, exception = get_validation_result(current_obs)
            if exception is not None:
                validation_msg += f" {exception}"
            elif step_idx > self.num_critique_steps:
                # if we have enough number of critique steps and the last output obs is valid
                break

            critique_outputs.append({"validation": validation_msg})
            critiquer_prompt = self._create_critiquer_prompt(
                simulator_inputs,
                sim_observation,
                critique_outputs,
            )

            if streaming_output:
                print(f"\n\n{validation_msg}\n\n")
                # for handler in callback_manager.handlers:
                #     getattr(handler, "on_text")("\n\n", verbose=self.verbose)

            crit_out = self.llm_critiquer.generate(
                [critiquer_prompt],
                stop=[
                    self.critique_prefix.format(step=step_idx + 1),
                    "Action:",
                    "Action Input:",
                ],
            )
            assert len(crit_out.generations) == 1
            # todo: this is for chat model
            crit_out = crit_out.generations[0][0].text
            # critique_outputs.append(crit_out)
            critique_outputs[-1]["critique"] = crit_out
            revised_output = self._extract_revised_observation_and_thought(
                crit_out, current_step=step_idx
            )
            current_obs = revised_output[0] if revised_output else current_obs

            log_output = sep + validation_msg + "\n" + crit_out
            if not streaming_output and not log_output.isspace():
                for handler in callback_manager.handlers:
                    getattr(handler, "on_tool_end")(log_output, verbose=self.verbose)

        # todo: extract sim_observation from sim_observation.log
        if revised_output is None:
            return sim_observation

        # todo: the correctness of logging need to be checked.
        logs = [sim_observation.log]
        for crit_dict in critique_outputs:
            logs.append(crit_dict["validation"] + "\n" + crit_dict["critique"])
        log_output_with_critique = sep.join(logs)

        critiqued_observation = SimulatedObservation(
            observation=revised_output[0],
            thought_summary=revised_output[1],
            log=log_output_with_critique,
        )
        # update log in observation
        return critiqued_observation

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentActionWithMetadata, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentActionWithMetadata, str]]]:
        """Override to use virtual tool execution and custom InvalidTool."""
        # Call the LLM to see what to do.
        print('===========intermediate_steps Start=========')
        for step in intermediate_steps:
            print(step[0].tool_input, step[0].is_read_only_action)
        print('===========intermediate_steps End=========')
        try:
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs
        )

        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.

        # Determine the action is a read-only operation or not here
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentActionWithMetadata]
        if isinstance(output, AgentAction) or isinstance(output, AgentActionWithMetadata):
            is_read_only_action = self.agent.is_read_only_action(output)
            output_with_metadata = AgentActionWithMetadata(
                tool=output.tool,
                tool_input=output.tool_input,
                log=output.log,
                is_read_only_action=is_read_only_action
            )
            actions = [output_with_metadata]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(
                    agent_action, verbose=self.verbose, color="green"
                )

            should_use_real_tool = self.should_use_real_tool(
                current_action=agent_action,
                intermediate_steps=intermediate_steps
            )
            if should_use_real_tool:
                # TODO: Select proper tools between real and virtual tools
                print('============Execute by Real Tools==========\n\n\n')
                tool = self.try_to_get_real_tool(agent_action.tool, name_to_tool_map)
                if tool:
                    return_direct = tool.return_direct
                    color = color_mapping[agent_action.tool]
                    tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                    if return_direct:
                        tool_run_kwargs["llm_prefix"] = ""
                    # We then call the tool on the tool input to get an observation
                    # KEY: Execute the real tools by the following line
                    observation = tool.run(
                        agent_action.tool_input,
                        verbose=self.verbose,
                        color=color,
                        callbacks=run_manager.get_child() if run_manager else None,
                        **tool_run_kwargs,
                    )
                else:
                    tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                    # * ====== ToolEmu: begin overwrite ======
                    observation = InvalidTool(available_tools=self.tool_names).run(
                        agent_action.tool,
                        verbose=self.verbose,
                        color=None,
                        callbacks=run_manager.get_child() if run_manager else None,
                        **tool_run_kwargs,
                    )
                    # * ====== ToolEmu: end overwrite ======
            else:
################################### Execute by Tool Emulator Start ##########################################################
                print('============Execute by Tool Emulator==========\n\n\n')
                # Otherwise we lookup the tool
                # TODO: Select proper tools between real and virtual tools
                if agent_action.tool in name_to_tool_map:
                    tool = name_to_tool_map[agent_action.tool]
                    return_direct = tool.return_direct
                    color = color_mapping[agent_action.tool]
                    tool_run_kwargs = self.agent.tool_run_logging_kwargs()

                    # We then call the llm to simulate the execution of the tool
                    empty_observation = ""  # for the current step
                    simulator_scratchpad = self._construct_simulator_scratchpad(
                        should_use_real_tool,
                        intermediate_steps + result + [(agent_action, empty_observation)]
                    )

                    # print(simulator_scratchpad)

                    full_inputs = {
                        "simulator_scratchpad": simulator_scratchpad,
                        "current_tool": agent_action.tool,
                        "current_tool_description": tool.description,
                        "toolkit_descriptions": self._get_current_toolkit_descriptions(
                            agent_action.tool
                        ),
                        **inputs,
                    }

                    # Decide to execute in real tools or emulated tools here
                    observation = run_with_input_validation(
                        self.llm_simulator_tool.run,
                        full_inputs,
                        tool,
                        agent_action.tool_input,
                        verbose=self.verbose,
                        color=color,
                        **tool_run_kwargs,
                    )

                    if isinstance(observation, str):
                        observation = SimulatedObservation(
                            observation=observation, thought_summary="", log=observation
                        )
                else:
                    tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                    observation_text = InvalidTool(available_tools=self.tool_names).run(
                        agent_action.tool,
                        verbose=self.verbose,
                        color=None,
                        **tool_run_kwargs,
                    )
                    observation = SimulatedObservation(
                        observation=observation_text,
                        thought_summary="",
                        log=observation_text,
                    )
    ################################### Execute by Tool Emulator End ##########################################################
            result.append((agent_action, observation))
        return result

    # async method hasn't been implemented yet 
    async def _atake_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Override to use virtual tool execution and custom InvalidTool."""
        # Call the LLM to see what to do.
        output = await self.agent.aplan(intermediate_steps, **inputs)
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if self.callback_manager.is_async:
                await self.callback_manager.on_agent_action(
                    agent_action, verbose=self.verbose, color="green"
                )
            else:
                self.callback_manager.on_agent_action(
                    agent_action, verbose=self.verbose, color="green"
                )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()

                # We then call the llm to simulate the execution of the tool
                empty_observation = ""  # for the current step
                simulator_scratchpad = self.agent._construct_scratchpad(
                    intermediate_steps + result + [(agent_action, empty_observation)],
                    include_last_observation=False,
                )
                # add prefix for generation
                simulator_scratchpad += self.generatetion_prefix

                full_inputs = {
                    "simulator_scratchpad": simulator_scratchpad,
                    "current_tool": agent_action.tool,
                    "current_tool_description": tool.description,
                    "toolkit_descriptions": self._get_current_toolkit_descriptions(
                        agent_action.tool
                    ),
                    **inputs,
                }

                # TODO: we may need to implement arun for the simulator tool
                observation = await run_with_input_validation(
                    self.llm_simulator_tool.arun,
                    full_inputs,
                    tool,
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    **tool_run_kwargs,
                )

                if isinstance(observation, str):
                    observation = SimulatedObservation(
                        observation=observation, thought_summary="", log=observation
                    )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = await InvalidTool(available_tools=self.tool_names).arun(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    **tool_run_kwargs,
                )

                observation = SimulatedObservation(
                    observation=observation, thought_summary="", log=observation
                )
            result.append((agent_action, observation))

        return result
    
    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentActionWithMetadata, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)
    
    def try_to_get_real_tool(self, tool_name: str, name_to_tool_map: Dict[str, BaseTool]) -> Optional[BaseTool]:
        return name_to_tool_map.get("""Real{tool_name}""".format(tool_name=tool_name), None)
    
    def should_use_real_tool(
        self,
        current_action: AgentActionWithMetadata,
        intermediate_steps: List[Tuple[AgentActionWithMetadata, str]]
    ) -> bool:
        if not current_action.is_read_only_action:
            return False
        
        if not all(step[0].is_read_only_action for step, _ in intermediate_steps):
            return False

        return True
