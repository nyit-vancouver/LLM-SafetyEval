const fs = require('fs')
const path = require('path')

function getAllFilesFromNestedDir(dirPath) {
  const files = fs.readdirSync(dirPath);
  const allFiles = [];

  for (let file of files) {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);
    if (stat.isDirectory()) {
      allFiles.push(...getAllFilesFromNestedDir(filePath));
    } else {
      allFiles.push(filePath);
    }
  }

  // only return .json files
  return allFiles.filter((file) => file.endsWith(".json"));
}

function mkdirDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

module.exports = {
    getAllFilesFromNestedDir,
    mkdirDir,
}
