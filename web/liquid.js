// Settings for Liquid Inference.
//
// The key goes to this package's server route and into a file in the ComfyUI
// user directory. These settings never call the setter that ComfyUI gives a
// renderer, so the key is not written to comfy.settings.json.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function getStatus() {
  const response = await api.fetchApi("/liquid-inference/status");
  return response.json();
}

async function post(path, body) {
  const response = await api.fetchApi(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const answer = await response.json();
  if (!response.ok) throw new Error(answer.error ?? response.statusText);
  return answer;
}

function keyText(status) {
  if (status.key_source === "environment") return "Set by LIQUID_API_KEY";
  return status.key_set ? "A key is saved" : "No key";
}

function baseUrlText(status) {
  if (status.base_url_source === "environment") return "Set by LIQUID_BASE_URL";
  return status.base_url_source === "settings" ? "Saved" : "Default";
}

function button(label, onClick) {
  const element = document.createElement("button");
  element.textContent = label;
  element.className = "p-button p-button-sm p-button-secondary";
  element.style.marginLeft = "4px";
  element.addEventListener("click", onClick);
  return element;
}

// Makes a field with Save and Clear buttons. `save` gets the text and
// returns the new status.
function field({ type, placeholder, describe, save, initial }) {
  const root = document.createElement("div");
  root.style.display = "flex";
  root.style.alignItems = "center";
  const input = document.createElement("input");
  input.type = type;
  input.autocomplete = "off";
  input.placeholder = placeholder;
  input.className = "p-inputtext p-component";
  input.style.width = "16rem";
  const note = document.createElement("span");
  note.style.marginLeft = "8px";
  note.style.opacity = "0.7";

  const show = (status) => {
    note.textContent = describe(status);
    initial?.(input, status);
  };
  const run = async (value) => {
    try {
      show(await save(value));
    } catch (error) {
      note.textContent = String(error.message ?? error);
    }
  };
  root.append(
    input,
    button("Save", () => run(input.value)),
    button("Clear", () => run("")),
    note,
  );
  getStatus().then(show, () => (note.textContent = "Status unknown"));
  return root;
}

app.registerExtension({
  name: "LiquidInference.Settings",
  settings: [
    {
      id: "LiquidInference.ApiKey",
      category: ["Liquid Inference", "Account", "API key"],
      name: "API key",
      tooltip:
        "Your Liquid Inference API key. It is saved in the ComfyUI user directory, not in workflows or images.",
      defaultValue: "",
      type: () =>
        field({
          type: "password",
          placeholder: "Paste a key, then Save",
          describe: keyText,
          save: (value) => post("/liquid-inference/key", { api_key: value }),
          initial: (input) => (input.value = ""),
        }),
    },
    {
      id: "LiquidInference.BaseUrl",
      category: ["Liquid Inference", "Account", "Base URL"],
      name: "Base URL",
      tooltip: "The router address. Clear it to use https://router.inference.ai.exchange/v1.",
      defaultValue: "",
      type: () =>
        field({
          type: "url",
          placeholder: "https://router.inference.ai.exchange/v1",
          describe: baseUrlText,
          save: (value) => post("/liquid-inference/base-url", { base_url: value }),
          initial: (input, status) => (input.value = status.base_url),
        }),
    },
  ],
});
