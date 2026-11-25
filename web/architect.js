// Settings for Architect Inference.
//
// The key goes to this package's server route and into a file in the ComfyUI
// user directory. These settings never call the setter that ComfyUI gives a
// renderer, so the key is not written to comfy.settings.json.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { priceBadge } from "./prices.js";

async function getStatus() {
  const response = await api.fetchApi("/architect-inference/status");
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
  if (status.key_expired) return "Key expired. Select Sign in again";
  if (status.key_source === "environment") {
    return `Set by ${status.key_env ?? "ARCHITECT_API_KEY"}`;
  }
  return status.key_set ? "A key is saved" : "No key";
}

function baseUrlText(status) {
  if (status.base_url_source === "environment") {
    return `Set by ${status.base_url_env ?? "ARCHITECT_BASE_URL"}`;
  }
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
function field({ type, placeholder, describe, save, initial, signIn = false }) {
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
  if (signIn) {
    const signInButton = button("Sign in", async () => {
      const popup = window.open("about:blank", "_blank");
      if (!popup) {
        note.textContent = "Allow popups for ComfyUI, then select Sign in again.";
        return;
      }
      popup.opener = null;
      signInButton.disabled = true;
      try {
        const { url } = await post("/architect-inference/sign-in", {});
        popup.location.replace(url);
        note.textContent = "Complete sign-in in the new page.";
        const deadline = Date.now() + 10 * 60 * 1000;
        while (Date.now() < deadline) {
          await new Promise((resolve) => setTimeout(resolve, 1500));
          const status = await getStatus();
          if (status.sign_in === "complete") { show(status); return; }
          if (status.sign_in === "failed") throw new Error(status.sign_in_error);
          if (popup.closed) throw new Error("Sign-in page closed. Select Sign in again if no key was saved.");
        }
        throw new Error("Sign-in expired. Select Sign in again.");
      } catch (error) {
        popup.close();
        note.textContent = String(error.message ?? error);
      } finally {
        signInButton.disabled = false;
      }
    });
    root.append(signInButton);
  }
  getStatus().then(show, () => (note.textContent = "Status unknown"));
  return root;
}

app.registerExtension({
  name: "ArchitectInference.Settings",
  settings: [
    {
      id: "ArchitectInference.ApiKey",
      category: ["Architect Inference", "Account", "API key"],
      name: "API key",
      tooltip:
        "Your Architect Inference API key. It is saved in the ComfyUI user directory, not in workflows or images.",
      defaultValue: "",
      type: () =>
        field({
          type: "password",
          signIn: true,
          placeholder: "Paste a key, then Save",
          describe: keyText,
          save: (value) => post("/architect-inference/key", { api_key: value }),
          initial: (input) => (input.value = ""),
        }),
    },
    {
      id: "ArchitectInference.BaseUrl",
      category: ["Architect Inference", "Account", "Base URL"],
      name: "Base URL",
      tooltip: "The router address. Clear it to use https://router.inference.ai.exchange/v1.",
      defaultValue: "",
      type: () =>
        field({
          type: "url",
          placeholder: "https://router.inference.ai.exchange/v1",
          describe: baseUrlText,
          save: (value) => post("/architect-inference/base-url", { base_url: value }),
          initial: (input, status) => (input.value = status.base_url),
        }),
    },
  ],
});

let priceDirectory = null;
let priceRequest = null;
let priceReadAt = 0;

function refreshPrices() {
  if (priceRequest || Date.now() - priceReadAt < 60000) return;
  priceReadAt = Date.now();
  priceRequest = api.fetchApi("/architect-inference/prices")
    .then((response) => {
      if (!response.ok) throw new Error("Price unavailable");
      return response.json();
    })
    .then((directory) => {
      priceDirectory = directory;
      app.graph?.setDirtyCanvas(true, true);
    })
    .catch(() => { priceDirectory = null; })
    .finally(() => { priceRequest = null; });
}

app.registerExtension({
  name: "ArchitectInference.Prices",
  nodeCreated(node) {
    if (!["ArchitectImage", "ArchitectVideo", "LiquidImage", "LiquidVideo"].includes(node.comfyClass)) return;
    node.addCustomWidget({
      name: "architect_price",
      type: "architect_price",
      options: { serialize: false },
      computeSize: (width) => [width, 26],
      draw(ctx, currentNode, width, y) {
        refreshPrices();
        const values = Object.fromEntries((currentNode.widgets ?? []).map((widget) => [widget.name, widget.value]));
        for (const input of currentNode.inputs ?? []) {
          if (input.link != null) values[input.name] = null;
        }
        const badge = priceBadge(priceDirectory, currentNode.comfyClass, values);
        ctx.save();
        // Dark-theme `danger` and `ink-2` colors from the Architect brand tokens.
        ctx.fillStyle = badge.low ? "#E3938B" : "#A3A8AE";
        ctx.font = "12px sans-serif";
        ctx.fillText(badge.text, 10, y + 17, width - 20);
        ctx.restore();
      },
    });
    node.setSize(node.computeSize());
    refreshPrices();
  },
  setup() {
    setInterval(refreshPrices, 60000);
  },
});
