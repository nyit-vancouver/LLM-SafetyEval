const { generateRiskDescription } = require("../risk_description_generator/");
const { GeneratedRiskDescription, Case } = require("../db/");
const {
  openaiGPT35Config,
  openaiGPT420240409Config,
} = require("../lib/request");

async function getCasesByIndustry(industry) {
  return Case.findAll({
    where: {
      industry: industry,
    },
  });
}

async function generateRiskDescriptionFromCases(caseObj, modelConfig) {
  const existingResult = await GeneratedRiskDescription.findOne({
    where: {
      case_dbid: caseObj.id,
      model_name: modelConfig.model,
    },
  });

  if (existingResult) {
    console.log(
      `Ignore case ${caseObj.id}, which has already been found in database with model ${modelConfig.model}`
    );
    return;
  }

  const { result, messages } = await generateRiskDescription(
    {
      trajectories: caseObj.contents,
    },
    modelConfig
  );

  await saveGeneratedRiskDescription({
    result,
    caseObj,
    messages,
    modelConfig,
  });
}

async function saveGeneratedRiskDescription({
  result,
  caseObj,
  messages,
  modelConfig,
}) {
  return GeneratedRiskDescription.create({
    case_dbid: caseObj.id,
    benchmark_id: caseObj.case_id,
    model_name: modelConfig.model,
    model_params: modelConfig.params,
    messages: messages,
    raw_result: JSON.stringify(result),
    risk_description: result.message.content,
  });
}

async function main(modelConfig, {} = {}) {
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

//   const categories = ["terminal"];

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
        await generateRiskDescriptionFromCases(caseObj, modelConfig);
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

// main(openaiGPT35Config)
//   .then(() =>
//     console.log(
//       `Risk description generation completed with model ${openaiGPT35Config.model}!`
//     )
//   )
//   .catch((err) => console.log(err));

main(openaiGPT420240409Config)
  .then(() =>
    console.log(
      `Risk description generation completed with model ${openaiGPT420240409Config.model}!`
    )
  )
  .catch((err) => console.log(err));