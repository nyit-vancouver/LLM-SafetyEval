const { getUserInstructionPrompt, getSystemPrompt } = require("./prompt");
const {
  makeRequestWithBackoff,
} = require("../lib/request");

// TODO: integrate with the initial context
async function generateRiskDescription({
  trajectories,
  initialContext = "",
} = {}, modelConfig) {
  const params = {
    messages: [
      {
        role: "system",
        content: getSystemPrompt({}),
      },
      {
        role: "user",
        content: getUserInstructionPrompt({
          trajectories,
        }),
      },
    ],
  };

  const res = await makeRequestWithBackoff(params, modelConfig);

  return {
    messages: params.messages,
    result: res.data.choices[0],
  };
}

module.exports = {
  generateRiskDescription,
};
