# Security and privacy policy

## Scope

VeriFace AI is a research prototype that runs locally. It has no server component, user accounts, network API or
telemetry. The Streamlit app processes uploads in memory and does not save them.

## Reporting a problem

Please report privately rather than in a public issue if you find any of these:

* a credential, personal data or a dataset image committed to this repository;
* a way for a crafted image or checkpoint to execute code or read files;
* any other security concern.

Use GitHub's **"Report a vulnerability"** (Security tab) on this repository, or contact the maintainer through the
GitHub profile [@Anshu-raj-co](https://github.com/Anshu-raj-co). Expect an acknowledgement within a few days.

## Things to know

* **Checkpoints are pickles.** `inference.py` loads checkpoints with `torch.load(..., weights_only=False)`, which can
  execute code embedded in a malicious file. **Only load checkpoints you trained yourself or received through a
  trusted channel**, and check them against the published SHA-256 in the README.
* **Do not upload images of other people** without their consent, including in issues and pull requests.
* **Results are not evidence.** Do not use the app's verdicts to accuse anyone or to make decisions about real people.
  See the limitations in the README.
