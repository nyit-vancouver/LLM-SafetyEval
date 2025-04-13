const axios = require("axios");

const openaiGPT35Config = {
  endpoint: "https://api.openai.com/v1/chat/completions",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
  },
  method: "post",
  model: "gpt-3.5-turbo",
  params: {
    temperature: 0,
    max_tokens: 4096,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  },
};

const openaiGPT420240409Config = {
  endpoint: "https://api.openai.com/v1/chat/completions",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
  },
  method: "post",
  model: "gpt-4-turbo-2024-04-09",
  params: {
    temperature: 0,
    max_tokens: 4096,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  },
};

async function makeRequestWithBackoff(
  params,
  modelConfig,
  requestConfig = { retries: 3, backoffDelay: 2000 },
) {
  try {
    const response = await axios[modelConfig.method](
      modelConfig.endpoint,
      {
        model: modelConfig.model,
        ...modelConfig.params,
        ...params,
      },
      {
        headers: modelConfig.headers,
      }
    );
    return response;
  } catch (error) {
    if (error.response && error.response.status === 429 && requestConfig.retries > 0) {
      // 429 is the HTTP status code for Too Many Requests
      // Wait for a random delay that increases exponentially with each retry
      const delay = Math.random() * requestConfig.backoffDelay;
      console.log(`Rate limit hit, retrying in ${delay}ms`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return makeRequestWithBackoff(params, modelConfig, {
        retries: requestConfig.retries - 1,
        backoffDelay: requestConfig.backoffDelay * 2,
      });
    } else {
      // If it's not a rate limit error or we ran out of retries, throw the error
      throw error;
    }
  }
}

module.exports = {
  openaiGPT35Config,
  openaiGPT420240409Config,
  makeRequestWithBackoff,
};
