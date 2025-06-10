const fs = require("fs");
const path = require("path");
const util = require("util");
const { cloneDeep } = require("lodash");

const { getAllFilesFromNestedDir, mkdirDir } = require('../lib/fs');

const CONVERTED_DIR = "./converted/";

const ACTION_TYPES = {
  JSON_RESULT: "jsonResult",
  ACTION_WITH_TOOL: "actionWithTool",
  PLAIN_TEXT: "plainText",
  NULL_ACTION: "nullAction",
};

function parseAction(actionStr) {
  // const pattern = /^([^:]+)(?::\s*(\{.*\}))?$/;

  const pattern = /^(.*?):\s*(\{.*\})$/;

  let action, actionName, type;

  if (!actionStr) {
    return {
      type: ACTION_TYPES.NULL_ACTION,
      actionName: "",
      action: "",
    };
  }

  if (actionStr.startsWith("{")) {
    return {
      type: ACTION_TYPES.JSON_RESULT,
      actionName: "",
      action: actionStr,
    };
  }

  const match = actionStr.match(pattern);

  if (match) {
    actionName = match[1];
    action = match[2];

    if (action.startsWith("{")) {
      return {
        type: ACTION_TYPES.JSON_RESULT,
        actionName,
        action,
      };
    }
  }

  return {
    type: ACTION_TYPES.PLAIN_TEXT,
    actionName: '',
    action: actionStr,
  };
}

function parseBenchmarkObject(obj) {
  const parsedObj = cloneDeep(obj);
  const { contents } = parsedObj;
  const parsedContents = contents.map((interactions) => {
    return interactions.map((interaction) => {
        if (interaction.role === 'agent') {
            interaction["parsedAction"] = parseAction(interaction.action);
        }

      return interaction;
    });
  });

  parsedObj.contents = parsedContents;

  return parsedObj;
}

function convertCases(cases) {
  return cases.map(parseBenchmarkObject);
}

function convertFileByPath(filePath, convertedDir = CONVERTED_DIR) {
  const filename = path.basename(filePath)
  const categoryDirPath = path.basename(path.dirname(filePath));
  const targetDir = path.join(convertedDir, categoryDirPath);

  // make sure the directory exists
  mkdirDir(targetDir);

  const fileContent = fs.readFileSync(filePath, "utf8");
  const json = JSON.parse(fileContent);

  const convertedResults = convertCases(json);
  const contentToWrite = JSON.stringify(convertedResults, null, 4)
  
  fs.writeFileSync(path.join(targetDir, filename), contentToWrite);
}

// prepare
mkdirDir(CONVERTED_DIR);
const benchmarkFiles = getAllFilesFromNestedDir("./benchmark/r-judge");

console.log("benchmarkFiles", benchmarkFiles);

benchmarkFiles.forEach((file) => {
  convertFileByPath(file);
});
