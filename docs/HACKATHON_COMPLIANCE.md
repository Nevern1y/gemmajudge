# AMD ACT II Track 3 Compliance

Verified on 2026-07-10 against the current official event page, the event-linked Participant
Submission Guide, lablab.ai Terms of Use and Code of Conduct, and the authenticated project
submission form.

## Authoritative Sources

- Event page: <https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii>
- Participant Submission Guide: <https://drive.google.com/file/d/1UGpOZiGGGBqQhGQxX7g19QAA-Dq9hPKk/view>
- Participation Terms: <https://lablab.ai/terms-of-use#16-participation-terms>
- Code of Conduct: <https://lablab.ai/code-of-conduct>
- General Rule Book: <https://lablab.ai/hackathon-rules>
- AMD Gemma 3 on MI300X guide: <https://rocm.blogs.amd.com/artificial-intelligence/deployingGemma-vllm/README.html>

## Submission Checklist

- [x] Registered Track 3 team and editable submission.
- [x] Public GitHub repository with runnable setup and usage instructions.
- [x] Required demo video uploaded; public guidance limits it to MP4, 5 minutes, and 300 MB.
- [x] Required PDF slide presentation uploaded; it can be replaced before the deadline.
- [x] Required live application fields completed with the Streamlit URL.
- [x] Dockerfile targets `linux/amd64`.
- [ ] Public `linux/amd64` container image published and placed in the Docker Image field.
- [x] MIT project license.
- [x] Third-party AI tooling disclosed in `AI_TOOL_DISCLOSURE.md` and submission copy.
- [x] AMD compute demonstrated with committed ROCm/vLLM artifacts on both historical W7900 and
  primary MI300X evidence runs.
- [x] Official-Gemma MI300X run captured with three model-generated attacks and committed
  evidence. This is scoped as a recorded demonstration, not a benchmark or human calibration.

Deadline: **2026-07-11 21:00 Kazakhstan Time (UTC+5), or 16:00 UTC**. The six-hour manual
submission path in the general rules is available only for a valid technical reason with
prior organizer or mentor approval; it is not a normal grace period.

## Conservative Resolution Of Rule Conflicts

The Track 3 guide says no Docker image is required and a live URL is optional. The current
event page says every submission must be containerized, while the authenticated form marks
Docker Image, Demo Application Platform, and Demo Application URL as required. GemmaJudge
therefore follows the stricter live-form requirements and provides all of them.

The guide lists 60-second container readiness, sub-30-second response time, English output,
and no hardcoded/cached answers as general rules. Even though Track 3 has no hidden request
harness or speed score, GemmaJudge follows them conservatively:

- OpenAI-compatible client calls default to a 25-second timeout and reject values above 30.
- The app and model outputs are English.
- The attacker generates unseen attacks; the final proof prompt contains no concrete
  few-shot questions to copy.
- Recorded and simulated public data are visibly labeled and are not presented as live
  hidden-input inference.
- The container has a health check and is tested for readiness before publishing.

## Model And Infrastructure Compliance

Track 3 allows open models, frameworks, and any tech stack, and requires demonstrated AMD
compute. The final new proof uses official `google/gemma-3-*` model IDs only after accepting
Google's Gemma Terms, served through vLLM on AMD Instinct MI300X with ROCm. Public mirrors
are not used for the submission proof, avoiding uncertainty about redistribution provenance.

Gemma weights remain under the Gemma Terms, not this repository's MIT license. Model weights,
tokens, adapters, and caches are not committed.

## Judging Scope

Track 3 is human-judged after automated pre-screening for AMD resource use and originality.
The pre-screen inspects the repository, PDF deck, and hosted URL, not the video. Public claims
therefore remain in those inspected artifacts and are scoped to measured evidence. The four
published judging areas are Creativity and Originality, Product/Market Potential,
Completeness, and Use of AMD Platforms; no official weights are published.
