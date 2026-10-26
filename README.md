# Liquid Inference for ComfyUI

This package adds three nodes to ComfyUI:

- **Liquid Image (paid API)** buys images.
- **Liquid Video (paid API)** buys a video.
- **Liquid Video Job** gets the video of a job again, with no new charge.

The paid nodes send a text prompt to
[Liquid Inference](https://inference.ai.exchange), an auction exchange for
inference. Sellers post offers for image and video models. The exchange gives
your request to the best offer that is under your price cap, and you pay for
the images or the video that you get.

> [!IMPORTANT]
> Each run of a paid node is a paid request to an external service. Your
> account pays for it. The prompt goes to the seller that wins the auction.

## Requirements

- ComfyUI 0.3.45 or newer.
- A Liquid Inference API key with the `spend` scope, and a balance on the
  account.

## Install

1. In ComfyUI, open **Manager**, then **Custom Nodes Manager**.
2. Search for "Liquid Inference".
3. Select **Install**, then restart ComfyUI.

To install by hand, clone this repository into `ComfyUI/custom_nodes/`, then
restart ComfyUI. The package has no Python dependencies to install.

## Set the API key

1. Open **Settings**, then **Liquid Inference**.
2. Paste your key into **API key**.
3. Select **Save**.

The key goes to the ComfyUI server. The server keeps it in a file in the
ComfyUI user directory, `user/__liquid_inference/api_key`. On Linux and
macOS, only your system user can read this file. ComfyUI does not serve
folders whose names start with `__`.

As an alternative, set the `LIQUID_API_KEY` environment variable before you
start ComfyUI. The environment variable has priority over the stored key.

The nodes have no input for the key. ComfyUI writes each input value into the
workflow file and into the metadata of each saved image. A key in an input
would thus go to each person who gets one of your images or workflows.

## Buy images

1. Double-click the canvas and search for "Liquid Image".
2. Write a prompt, and select a model.
3. Connect `images` to **Save Image** or to a different image node.
4. Queue the workflow.

The `example_workflows` folder has five workflows: text to image, text to
image with an upscale, two models side by side, text to video and image to
video. In ComfyUI, open them from **Templates**.

### Inputs

| Input | Description |
| --- | --- |
| `prompt` | The text that describes the image. |
| `model` | An image model that a seller offers now. The node gets the list from the exchange and refreshes it each minute. |
| `size` | `auto`, or a size in pixels. `auto` lets each seller use its default size, so more offers can take the request. An explicit size goes only to the sellers that declared it. |
| `n` | The number of images, from 1 to 4. |
| `cap_usd` | The most that one run of the node can cost, in USD. |

The cap applies to each request, not to the workflow. A workflow with three
Liquid nodes and a cap of 0.05 can cost up to 0.15 USD for each run. If you
queue the workflow ten times, each run is a new request with its own cap.

ComfyUI does not run a node again when its inputs did not change. Thus a
second queue of the same workflow does not buy the same images again. Change
an input to buy new images.

### Outputs

| Output | Description |
| --- | --- |
| `images` | The images, as a ComfyUI `IMAGE` batch. |
| `job_id` | The exchange's ID for the job. Give this ID to support with each question about a charge. |
| `receipt` | JSON with the job ID, the model, the seller, the charge in USD, your cap and the node version. |

## Buy a video

1. Double-click the canvas and search for "Liquid Video".
2. Write a prompt, and select a model.
3. Connect `video` to **Save Video**.
4. Queue the workflow.

The node buys the video with one request. Then it reads the status of the
job every 5 seconds, and it shows the status on the node. When the job is
complete, the node gets the file from the exchange. The node waits for 30
minutes at most.

### Inputs

| Input | Description |
| --- | --- |
| `prompt` | The text that describes the video. |
| `model` | A video model that a seller offers now. The node gets the list from the exchange and refreshes it each minute. |
| `size` | `auto`, or a size in pixels. `auto` lets each seller use its default size. An explicit size goes only to the sellers that declared it. |
| `seconds` | `auto`, or the length of the video in seconds. `auto` uses the shortest length that a seller declares for the model. |
| `cap_usd` | The most that one run of the node can cost, in USD. |
| `start_image` | Optional. The first frame of the video, as one image. Only the models that read an image input can take it. Use an image with the shape of `size`, because a seller can crop a different shape. |
| `retention_policy` | `none` (the default) excludes providers that can keep your content. `allowed` permits those providers. |

Select `allowed` only if you accept that the provider can keep your prompt,
start image and video. The node sends this choice as
`exchange.retention_policy`. Old workflows use `none` until you change it.

A video is priced for each second. Thus the charge is the price of the
length that you ask for, and the cap must be at least that price.

### Outputs

| Output | Description |
| --- | --- |
| `video` | The video file, as a ComfyUI `VIDEO`. Connect it to **Save Video**, or to **Get Video Components** for the frames. |
| `job_id` | The exchange's ID for the job. |
| `receipt` | JSON with the job ID, the model, the seller, the charge and the cap in USD, the length, the size, the expiry of the file and the node version. |

### Stop a run

A stop in ComfyUI stops the wait, not the job. The exchange has no route to
cancel a video job. The seller continues, and the exchange charges for the
video when the job is complete. The error gives the job ID.

To get the video of a job that you paid for, use the **Liquid Video Job**
node:

1. Add a **Liquid Video Job** node.
2. Paste the job ID into `job_id`.
3. Connect `video` to **Save Video**, and queue the workflow.

This node buys nothing. It needs a key that can read the account's jobs. The
seller keeps the file until the time in `expires_at` in the receipt.

## Privacy

- The prompt goes to the Liquid Inference router, and then to the seller
  that wins the auction. The seller can be a different company for each run.
- The images come back from the seller. If the seller returns a link and not
  the image bytes, the node gets the image from that link. The node does not
  send your key to that link.
- A video comes through the exchange. The exchange does not keep the file.
- Anyone who can open your ComfyUI server can queue paid runs with your key.
  Do not start ComfyUI with `--listen` on a network that you do not trust.

## Retries

The paid nodes never send a purchase again by themselves. A second request
could buy the images or the video a second time. After the purchase of a
video, the node tries a failed status read or download again, because these
requests buy nothing.

If the connection stops before the answer to a purchase comes back, the node
shows an error that says that the outcome is unknown. The job possibly ran, and your
account possibly paid for it. The error gives an `Idempotency-Key`. Give it
to support to find the job.

## Troubleshoot

| Error | What to do |
| --- | --- |
| No Liquid Inference API key | Set the key in Settings, or set `LIQUID_API_KEY`, then try again. |
| `no_eligible_offer` with `retention_policy` | For a video, select `retention_policy = allowed` only if you accept content retention. Otherwise, select a different model. |
| HTTP 401 | The exchange does not know the key. Make a new key and save it. |
| HTTP 402 `insufficient_balance` | Add funds to your account. |
| HTTP 400 `unsupported_parameter` | No offer for the model accepts a value that you sent, usually `size`. Set `size` to `auto`, or select a different model. |
| HTTP 503 `no_eligible_offer` with `cap_exceeds_buyer_cap` | Each offer costs more than `cap_usd`. Increase `cap_usd`. |
| HTTP 503 `no_eligible_offer`, other reasons | No seller can take the request now. Try again later, or select a different model. |
| HTTP 404 `model_not_found` | No seller lists the model now. Reload the page to get the current model list, then select a model. |
| HTTP 503 `model_unavailable` | The model has sellers, but none of them can take a job now. Try again in a minute, or reload the page and select a different model. |
| HTTP 503 `provider_failed` | The seller failed before it made an image, and you were not charged. Run the node again. If no other seller has the model, the next run gets `model_unavailable`. |
| HTTP 502 or 504 with a web page | A proxy between ComfyUI and the exchange answered. Look at your receipts before you run the node again. |
| The model list shows "no image models" or "no video models" | ComfyUI could not get the model list. Check the network and the base URL, then reload the page. |
| A video job failed | The seller did not make the video, and you were not charged. Run the node again, or select a different model. |
| The node stopped waiting for a video job | The job continues, and the exchange charges for it. Put the job ID in a **Liquid Video Job** node to get the video. |
| HTTP 410 `video_expired` | The seller deleted the file. Look at the job's receipt for the charge. |
| The outcome is unknown | Do not run the node again at once. Look at your receipts, or give the `Idempotency-Key` to support. |

Each refusal from the exchange includes the code and the reasons that the
offers were excluded. A refusal before the auction costs nothing.

## Base URL

The node sends requests to `https://router.inference.ai.exchange/v1`. To use a
different router, set **Base URL** in Settings or set the `LIQUID_BASE_URL`
environment variable.

## Development

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv run pytest            # unit tests, with a local stub router
uv run ruff check
uv run ruff format --check
uv run mypy
```

The smoke test installs a pinned ComfyUI release and CPU torch, starts
ComfyUI against the stub router, and runs each example workflow. Then it
makes sure that no saved image, no workflow and no server answer holds the
key. The first run takes some minutes.

```bash
uv run pytest -m smoke
```

To keep the ComfyUI installation for the next run, set `LIQUID_SMOKE_DIR` to
a folder.

## Contribute

This repository is a copy of the source, which is not public. Each change
comes from the source, and a tag `v<version>` publishes that version to the
Comfy Registry. Open an issue or a pull request here. A maintainer applies
an accepted change in the source, and the next copy includes it. Do not
expect a merge of the pull request itself.

## License

MIT. See [LICENSE](LICENSE).
