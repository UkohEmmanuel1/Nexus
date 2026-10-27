# Releasing Vonk CLI

The source packages are version 0.3.0. Public installation works only after
the respective registry accepts the upload. A Git push is not a registry release.

Target commands after successful publication:

```bash
npm install -g vonk
pip install vonk-ai-terminal
vonk
```

`pip install vonk` belongs to an unrelated PyPI project. The agreed Python
distribution name is `vonk-ai-terminal`; its executable is still `vonk`.
The npm name `vonk` currently returns an unpublished-package response. That is
not proof of permission to claim the name; the publishing account must be allowed
to publish it. Do not silently change to another package name if the registry
rejects ownership.

## Before publishing

1. Keep npm/Python package versions and CLI VERSION values identical.
2. Ensure CI passes, including Windows/Linux interactive workspace and real
   subprocess MCP integration tests.
3. Review the package README, model access notes, and MCP consent behavior.
4. Run the Publish Vonk CLI workflow with `build-only` to produce checked wheel,
   source distribution, and npm tarball artifacts.

For local validation:

```bash
python -m pip install build twine
python -m build --outdir dist/python ./packages/python-cli
python -m twine check dist/python/*
npm pack ./packages/npm-cli --pack-destination dist
```

## Registry authorization

For PyPI, configure a pending Trusted Publisher for `vonk-ai-terminal` using:
owner `UkohEmmanuel1`, repository `Vonk`, workflow `publish-cli.yml`, environment
`pypi`. The workflow publishes via OIDC; no PyPI password needs to enter chat or
Git. See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

For npm, use an account authorized for `vonk` and a suitable publishing token in
the `npm` GitHub environment's `NPM_TOKEN` secret. Configure environment reviewers
if desired. Do not commit registry tokens. Alternatively, log in locally with
`npm login` and publish the reviewed tarball yourself; 2FA may be required.

Once authorization is configured, dispatch the workflow for the intended
registry, or `both`. It revalidates on Windows and Linux before publishing.
Never overwrite a version: bump versions before releasing changes.

## Verify the actual public release

```bash
npm view vonk version
python -m pip index versions vonk-ai-terminal
```

In clean environments, install from the public registries and test `vonk`,
`vonk chat`, model selection, skill context, and a trusted MCP server. Only then
announce that public registry installation is live. Source/tarball installation
is a valid preview but is not proof of registry publication.

Free local models need Ollama and downloaded weights. Hosted free tiers have
provider-controlled access and quotas. Vonk currently does not subsidize an
anonymous public inference service. Core chat is read-only; MCP calls are direct
user-authorized actions rather than an autonomous coding-agent loop.
