# Letting Claude Cook

Course materials for *Letting Claude Cook*, taught at Harvard.

## How to use this repo

This is the course's public materials repo. You will keep your own work in a
personal fork, and you'll pull updates from this repo as new material is posted.

### One-time setup

1. **Fork this repo.** Click the **Fork** button at the top right of
   https://github.com/dfink/letting-claude-cook. This creates a copy under your
   own GitHub account.

2. **Clone your fork** to your computer (replace `YOUR-USERNAME`):

   ```bash
   git clone https://github.com/YOUR-USERNAME/letting-claude-cook.git
   cd letting-claude-cook
   ```

3. **Add the course repo as `upstream`** so you can pull new material:

   ```bash
   git remote add upstream https://github.com/dfink/letting-claude-cook.git
   ```

### Getting new course material

Whenever new material is posted, run:

```bash
git pull upstream main
```

### Saving your work

Commit and push to your fork as you go:

```bash
git add .
git commit -m "Describe what you did"
git push
```

### Submitting or asking for feedback

Open a **pull request** from your fork to this repo. Put your name and the
assignment in the PR title. We'll review it there.

## Course staff

Instructor: Doug Finkbeiner
