---
name: hackathon-deploy
description: >-
  Use this skill when the user wants to deploy the project, push to production,
  or prepare a demo build. Handles build verification and deployment steps.
---

# Hackathon Deploy

## Steps

1. Run lint: `npm run lint`
2. Run build: `npm run build`
3. Verify `dist/` directory was created with no errors.
4. If tests exist, run them: `npm test`
5. Commit and push to `main` branch.
6. Report the deployment status and any build warnings.
