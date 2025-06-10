const util = require("util");

const {
  makeRequestWithBackoff,
  openaiGPT35Config,
  openaiGPT420240409Config,
} = require("../lib/request");
const {
  Case,
  RiskDescriptionExperiment,
  GeneratedRiskDescription,
} = require("../db/");
const {
  getSystemPrompt,
  getUserInstructionPrompt,
} = require("../evaluator/prompt");

function parseSocreFromEvaluationResult(text) {
  const regex = /Overall Quantitative Score:\s*(\d+)/;
  const match = text.match(regex);

  if (match) {
    return parseFloat(match[1]);
  }

  return null;
}

async function getCasesByIndustry(industry) {
  return Case.findAll({
    where: {
      industry: industry,
    },
  });
}

async function runCaseExperimentWithCustomRiskDescription(
  caseObj,
  riskDescriptionModel,
  modelConfig
) {
  const customRiskDescription = await GeneratedRiskDescription.findOne({
    where: {
      case_dbid: caseObj.id,
      model_name: riskDescriptionModel,
    },
  });

  if (!customRiskDescription) {
    console.error(
      `Custom risk description for case ${caseObj.id} with model ${riskDescriptionModel} not found. Ignore case ${caseObj.id}`
    );
    return;
  }

  const existingResult = await RiskDescriptionExperiment.findOne({
    where: {
      case_dbid: caseObj.id,
      model_name: modelConfig.model,
      is_custom_risk_description: true,
      risk_description_id: customRiskDescription.id,
    },
  });

  if (existingResult) {
    console.log(
      `Ignore case ${caseObj.id}, which has already been evaluated with model ${modelConfig.model}`
    );
    return;
  }

  const params = {
    messages: [
      {
        role: "system",
        content: getSystemPrompt({}),
      },
      {
        role: "user",
        content: getUserInstructionPrompt({
          riskDescription: customRiskDescription.risk_description,
          trajectories: caseObj.contents,
        }),
      },
    ],
  };

  const res = await makeRequestWithBackoff(params, modelConfig);

  return writeResultWithCustomRiskDescriptionToDB(
    caseObj,
    customRiskDescription,
    params.messages,
    res.data.choices[0],
    modelConfig
  );
}

async function writeResultWithCustomRiskDescriptionToDB(
  caseObj,
  riskDescription,
  messages,
  result,
  modelConfig
) {
  const dataToSave = {
    case_dbid: caseObj.id,
    benchmark_id: caseObj.case_id,
    model_name: modelConfig.model,
    is_custom_risk_description: true,
    risk_description_id: riskDescription.id,
    risk_description: riskDescription.risk_description,
    initial_context: "",
    model_params: modelConfig.params,
    quantitative_score: parseSocreFromEvaluationResult(result.message.content),
    messages: messages,
    raw_result: JSON.stringify(result),
    is_trajectory_only: false,
  };

  return RiskDescriptionExperiment.create(dataToSave);
}

async function runCaseExperimentWithOriginalRiskDescription(
  caseObj,
  modelConfig,
  isTrajectoryOnly = false
) {
  // check if the case has already been evaluated in RiskDescriptionExperiment by dbid and model_name
  const existingResult = await RiskDescriptionExperiment.findOne({
    where: {
      case_dbid: caseObj.id,
      model_name: modelConfig.model,
      is_trajectory_only: isTrajectoryOnly,
      is_custom_risk_description: false,
    },
  });


  if (existingResult) {
    console.log(
      `Ignore case ${caseObj.id}, which has already been evaluated with model ${modelConfig.model}`
    );
    return;
  }

  const params = {
    messages: [
      {
        role: "system",
        content: getSystemPrompt({}),
      },
      {
        role: "user",
        content: getUserInstructionPrompt({
          riskDescription: isTrajectoryOnly ? "" : caseObj.risk_description,
          trajectories: caseObj.contents,
        }),
      },
    ],
  };

  const res = await makeRequestWithBackoff(params, modelConfig);

  return writeResultWithOriginalRiskDescriptionToDB(
    caseObj,
    params.messages,
    res.data.choices[0],
    modelConfig,
    isTrajectoryOnly
  );
}

async function writeResultWithOriginalRiskDescriptionToDB(
  caseObj,
  messages,
  result,
  modelConfig,
  isTrajectoryOnly
) {
  const dataToSave = {
    case_dbid: caseObj.id,
    benchmark_id: caseObj.case_id,
    model_name: modelConfig.model,
    is_custom_risk_description: false,
    risk_description_id: null,
    risk_description: caseObj.risk_description,
    initial_context: "",
    model_params: modelConfig.params,
    quantitative_score: parseSocreFromEvaluationResult(result.message.content),
    messages: messages,
    raw_result: JSON.stringify(result),
    is_trajectory_only: isTrajectoryOnly,
  };

  return RiskDescriptionExperiment.create(dataToSave);
}

async function main(
  modelConfig,
  { isTrajectoryOnly = false, riskDescriptionModel = "" } = {}
) {
  const categories = [
    "bitcoin",
    "moneymanagement",
    "webshop",
    "chatbot",
    "medical",
    "household",
    "trafficdispatch",
    "mobile",
    "windows",
    "terminal",
    "software",
    "security",
    "code_agentmonitor",
    "mail",
    "productivity",
    "socialapp",
    "webbrowser",
    "websearch",
  ];

  // const categories = ['terminal']

  const failedCases = [];

  for (let cidx = 0; cidx < categories.length; cidx++) {
    const cases = await getCasesByIndustry(categories[cidx]);

    for (let i = 0; i < cases.length; i++) {
      const caseObj = cases[i];
      console.log(
        `Run ${caseObj.category}-${caseObj.industry} with case ${
          caseObj.id
        } and model ${modelConfig.model} (${i + 1}/${cases.length})`
      );
      try {
        if (riskDescriptionModel) {
          await runCaseExperimentWithCustomRiskDescription(
            caseObj,
            riskDescriptionModel,
            modelConfig
          );
        } else {
          await runCaseExperimentWithOriginalRiskDescription(
            caseObj,
            modelConfig,
            isTrajectoryOnly
          );
        }
      } catch (err) {
        console.error(err);
        failedCases.push(caseObj);
        console.log(
          `Run ${caseObj.category}-${caseObj.industry} with case ${caseObj.id} and model ${modelConfig.model} failed (current failed counts = ${failedCases.length})`
        );
      }
    }
  }

  console.log(
    "Failed cases:",
    failedCases
      .map((c) => `${c.category}-${c.industry}-${c.id}-${c.case_id}`)
      .join(",\n")
  );
}

main(openaiGPT35Config, { isTrajectoryOnly: false })
  .then(() => console.log("Done!"))
  .catch((err) => console.log(err));

// main(openaiGPT35Config, { isTrajectoryOnly: true })
//   .then(() => console.log("Done!"))
//   .catch((err) => console.log(err));
// main(openaiGPT35Config, {
//   isTrajectoryOnly: false,
//   riskDescriptionModel: "gpt-3.5-turbo",
// })
//   .then(() => console.log("Done!"))
//   .catch((err) => console.log(err));

// main(openaiGPT35Config, {
//   isTrajectoryOnly: false,
//   riskDescriptionModel: "gpt-4-turbo-2024-04-09",
// })
//   .then(() => console.log("Done!"))
//   .catch((err) => console.log(err));

// main(openaiGPT420240409Config, { isTrajectoryOnly: false })
//   .then(() => console.log("Done!"))
//   .catch((err) => console.log(err));

// main(openaiGPT420240409Config, { isTrajectoryOnly: true })
//   .then(() => console.log("Done!"))
//   .catch((err) => console.log(err));
