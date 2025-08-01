# A Context-Aware LLM-Based Action Safety Evaluator for Automation Agents

While rapid advancements in Large Language Models (LLMs) have made the deployment of automation agents, such as AutoGPT and Open Interpreter, increasingly feasible, they also introduce new security challenges. We contribute to the field of agentic AI by proposing a context-aware LLM-based safety evaluator to assess the security implications of actions and instructions generated by LLM-based automation agents prior to execution in real environments. This approach does not require an expensive sandbox, prevents possible system damage from execution, and gathers additional runtime-related information for risk assessment. Our evaluator utilizes a semi-emulator tool designed for local real-time usage. Experiments show that using environmental feedback from readonly actions can help generate more accurate risk descriptions for the safety evaluator.

# Paper

C. Lin, A. Milani Fard, ["A Context-Aware LLM-Based Action Safety Evaluator for Automation Agents”](https://people.ece.ubc.ca/aminmf/LLM_SafetyEval_CanadianAI2025.pdf), The 38th Canadian Conference on Artificial Intelligence (Canadian AI), 2025.


# Citation

```
@article{Lin2025Context,
	author = {Lin, Chia-Hao and Milani Fard, Amin},
	journal = {Proceedings of the Canadian Conference on Artificial Intelligence},
	year = {2025},
	note = {https://caiac.pubpub.org/pub/63wkp5l0},
	publisher = {Canadian Artificial Intelligence Association (CAIAC)},
	title = {A {Context}-{Aware} {LLM}-{Based} {Action} {Safety} {Evaluator} for {Automation} {Agents}},
}
```
