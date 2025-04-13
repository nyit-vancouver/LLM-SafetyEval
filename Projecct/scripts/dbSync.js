const { sequelize } = require("../db/");

async function init() {
  await sequelize.sync();
}

init()
  .then(() => console.log("Database soft synced!"))
  .catch((err) => console.log(err));
