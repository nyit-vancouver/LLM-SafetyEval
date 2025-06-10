const fs = require('fs')
const path = require('path')

const { sequelize, Case } = require("../db/");
const { getAllFilesFromNestedDir, mkdirDir } = require('../lib/fs');

async function importBenchmarkData() {
    const files = getAllFilesFromNestedDir('./converted/');
    
    files.forEach(async (filePath) => {
        const cases = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        const category = path.basename(path.dirname(filePath));
        const fullName = path.basename(filePath);
        const extension = path.extname(filePath);
        const industry = fullName.replace(extension, '');

        cases.forEach(async (c) => {
            await Case.create({
                case_id: c.id,
                scenario: c.scenario,
                profile: c.profile,
                goal: c.goal,
                contents: c.contents,
                label: c.label,
                risk_description: c.risk_description,
                category: category,
                industry: industry,
                file_path: filePath,
            });
        })
    });
}

async function init() {
  await sequelize.sync({ force: false });
  await importBenchmarkData();
}

init()
  .then(() => console.log("Database & tables created!"))
  .catch((err) => console.log(err));
