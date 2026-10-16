# Contributing to VeriFace AI

Thanks for your interest. VeriFace AI is a research project, so the most valuable contributions are the ones that make
its claims **more accurate**, not more impressive.

## Useful contributions

* **Failure cases:** describe the input type, the verdict you got and what you expected. Do **not** attach photos of
  real people; describe them, or use images you own and are allowed to share.
* **Documentation errors:** a number in the README that does not match the artifact it cites, a broken link, or a
  command that does not work.
* **Bug fixes** in the pipeline, app or evaluation code.
* **New evaluations** on independently sourced data, especially generators and sources that are absent from the
  training data.

## Ground rules

1. **Never commit data or weights.** That includes images, face crops, `cleaned_metadata.csv`, per-image prediction
   files, `*.pth` and other checkpoints, and feature arrays. `.gitignore` blocks the common cases; check
   `git status` before you commit anyway.
2. **No secrets.** The project needs no API keys. If you add a feature that needs one, read it from an environment
   variable and document it in a `.env.example` with placeholder values only.
3. **Keep the label convention:** REAL = 0, FAKE = 1 (positive class). `inference.load_checkpoint` and
   `evaluate.load_model` refuse other checkpoints on purpose.
4. **Keep inference and evaluation identical.** Preprocessing lives in `face_utils.crop_face_square` and
   `train.get_eval_transforms`; the app must keep using `inference.py`.
5. **Research integrity:**
   * Select models, thresholds and temperatures on the **validation** split only.
   * Do not tune anything on the test split. State clearly that it has already been examined (`RESEARCH.md` §10).
   * Do not overwrite earlier reports in `research_runs/`. Add a new dated run directory, and record corrections
     alongside the original.
   * Report negative results.

## Workflow

```powershell
git checkout -b my-change
python model.py         # quick self-check
python face_utils.py    # quick self-check
# make your change, update docs if behaviour or numbers change
git status              # confirm no data, weights or logs are staged
```

Open a pull request that describes what changed, how you checked it, and any effect on reported metrics.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
