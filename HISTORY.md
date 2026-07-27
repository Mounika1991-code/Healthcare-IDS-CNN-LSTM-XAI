HISTORY
=======

On 2026-07-27 the repository history was rewritten to correct commit author information and remove an unintended contributor entry.

Why this was done
- Two commits were authored with the email `ashwinireddy@tkrcet.com` (GitHub user `eashwini03`).
- These commits were rewritten to attribute them to `Mounika1991-code <mkamigari@gmail.com>`.

Important notes for collaborators
- This rewrite changed commit SHAs and replaced the remote branch with a force-push.
- If you previously cloned or pulled this repository, you must re-sync to avoid conflicts.

To re-sync safely, run:

```
# discard local commits and match the remote exactly
git fetch origin
git reset --hard origin/master
```

Or simply reclone the repository:

```
git clone https://github.com/Mounika1991-code/Healthcare-IDS-CNN-LSTM-XAI.git
```

If you had local branches with work you need to preserve, create patches or rebase them onto the new `master`.

Contact
- If you need assistance re-syncing, open an issue in the repository or contact the repository owner.
