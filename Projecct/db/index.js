const { Sequelize, DataTypes, Model } = require("sequelize");

const sequelize = new Sequelize({
  dialect: "sqlite",
  // storage: "db.sqlite",
  storage: "db20240504.sqlite",
});

class Case extends Model {}
Case.init(
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    case_id: DataTypes.INTEGER,
    scenario: DataTypes.STRING,
    profile: DataTypes.TEXT,
    goal: DataTypes.TEXT,
    contents: {
      type: DataTypes.TEXT,
      get() {
        const rawValue = this.getDataValue("contents");
        return rawValue ? JSON.parse(rawValue) : null;
      },
      set(value) {
        this.setDataValue("contents", JSON.stringify(value));
      },
    },
    label: DataTypes.INTEGER,
    risk_description: DataTypes.TEXT,
    category: DataTypes.STRING,
    industry: DataTypes.STRING,
    file_path: DataTypes.TEXT,
  },
  { sequelize, modelName: "Case" }
);

class RiskDescriptionExperiment extends Model {}
RiskDescriptionExperiment.init(
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    case_dbid: DataTypes.INTEGER,
    benchmark_id: DataTypes.INTEGER,
    model_name: DataTypes.STRING,
    is_custom_risk_description: DataTypes.BOOLEAN,
    risk_description_id: DataTypes.INTEGER,
    risk_description: DataTypes.TEXT,
    initial_context: DataTypes.TEXT,
    is_trajectory_only: DataTypes.BOOLEAN,
    model_params: {
      type: DataTypes.TEXT,
      get() {
        const rawValue = this.getDataValue("model_params");
        return rawValue ? JSON.parse(rawValue) : null;
      },
      set(value) {
        this.setDataValue("model_params", JSON.stringify(value));
      },
    },
    quantitative_score: DataTypes.FLOAT,
    messages: {
      type: DataTypes.TEXT,
      get() {
        const rawValue = this.getDataValue("messages");
        return rawValue ? JSON.parse(rawValue) : null;
      },
      set(value) {
        this.setDataValue("messages", JSON.stringify(value));
      },
    },
    raw_result: DataTypes.TEXT,
  },
  { sequelize, modelName: "RiskDescriptionExperiment" }
);

class GeneratedRiskDescription extends Model {}
GeneratedRiskDescription.init(
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    case_dbid: DataTypes.INTEGER,
    benchmark_id: DataTypes.INTEGER,
    model_name: DataTypes.STRING,
    model_params: {
      type: DataTypes.TEXT,
      get() {
        const rawValue = this.getDataValue("model_params");
        return rawValue ? JSON.parse(rawValue) : null;
      },
      set(value) {
        this.setDataValue("model_params", JSON.stringify(value));
      },
    },
    messages: {
      type: DataTypes.TEXT,
      get() {
        const rawValue = this.getDataValue("messages");
        return rawValue ? JSON.parse(rawValue) : null;
      },
      set(value) {
        this.setDataValue("messages", JSON.stringify(value));
      },
    },
    raw_result: DataTypes.TEXT,
    risk_description: DataTypes.TEXT,
  },
  { sequelize, modelName: "GeneratedRiskDescription" }
);

module.exports = {
  sequelize,
  Case,
  RiskDescriptionExperiment,
  GeneratedRiskDescription,
};
