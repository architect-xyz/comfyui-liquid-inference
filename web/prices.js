export function priceBadge(directory, kind, values) {
  const video = kind === "LiquidVideo";
  const model = (directory?.[video ? "videos" : "images"] ?? [])
    .find((entry) => entry.id === values.model);
  const rate = model?.[video ? "rate_per_video_second_usd" : "rate_per_image_usd"];
  const units = video
    ? values.seconds === "auto"
      ? Math.min(...(model?.seconds ?? []))
      : Number(values.seconds)
    : Number(values.n);
  const cap = Number(values.cap_usd);
  const capText = Number.isFinite(cap) && cap > 0 ? `$${cap}` : "unknown";
  if (!(rate > 0) || !Number.isFinite(units) || units <= 0) {
    return { text: `Price unavailable · cap ${capText}`, low: false };
  }
  const estimate = rate * units;
  return {
    text: `from $${estimate.toFixed(3)}${video ? ` / ${units} s` : units === 1 ? " / image" : ` / ${units} images`} + fee · cap ${capText}`,
    low: Number.isFinite(cap) && cap > 0 && cap < estimate,
  };
}
