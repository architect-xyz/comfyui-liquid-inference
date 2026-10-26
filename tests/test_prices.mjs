import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const source = await readFile(new URL("../web/prices.js", import.meta.url), "utf8");
const { priceBadge } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const directory = {
  images: [{ id: "image", rate_per_image_usd: 0.04 }],
  videos: [{ id: "video", rate_per_video_second_usd: 0.05, seconds: [8, 4] }],
};
test("image count changes the estimate and low-cap warning", () => {
  const badge = priceBadge(directory, "LiquidImage", { model: "image", n: 2, cap_usd: 0.05 });
  assert.equal(badge.low, true);
  assert.match(badge.text, /\$0.080/);
  assert.match(badge.text, /\+ fee/);
});
test("auto video duration uses the shortest declared duration", () => {
  const badge = priceBadge(directory, "LiquidVideo", { model: "video", seconds: "auto", cap_usd: 0.3 });
  assert.equal(badge.low, false);
  assert.match(badge.text, /\$0.200/);
  assert.match(priceBadge(directory, "LiquidVideo", { model: "video", seconds: "8", cap_usd: 0.3 }).text, /\$0.400/);
});
test("a missing price or linked input gives an unknown estimate", () => {
  assert.equal(priceBadge(directory, "LiquidImage", { model: "missing", n: 1, cap_usd: 1 }).low, false);
  assert.match(priceBadge(directory, "LiquidImage", { model: "image", n: null, cap_usd: 1 }).text, /unavailable/);
  assert.match(priceBadge(directory, "LiquidVideo", { model: "video", seconds: null, cap_usd: 1 }).text, /unavailable/);
});
