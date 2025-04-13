function getScratchpadStringFromTrajectories(trajectories) {
    return trajectories
    .map((trajectory) => {
      let scratchpad = "";
    
      for (let i = 0; i < trajectory.length; i++) {
        const interaction = trajectory[i];
    
        if (interaction.role === "user") {
          scratchpad += `User Input: ${interaction.content}\n`;
        }
    
        if (interaction.role === "agent") {
          const { parsedAction } = interaction;
    
          if (parsedAction.actionName) {
            scratchpad += `Action: ${parsedAction.actionName}\n`;
          }
    
          if (parsedAction.action) {
            if (i >= trajectory.length - 2 && !parsedAction.actionName) {
              scratchpad += `Final Answer: ${interaction.action}\n`;
            } else {
              scratchpad += `Action Input: ${parsedAction.action}\n`;
            }
          }
    
          if (interaction.thought) {
            scratchpad += `Thought: ${interaction.thought}\n`;
          }
        }
    
        if (interaction.role === "environment") {
          if (interaction.content) {
            scratchpad += `Observation: ${interaction.content}\n`;
          }
        }
      }
    
      return scratchpad;
    })
    .join("\n\n");
}

module.exports = {
    getScratchpadStringFromTrajectories
}
