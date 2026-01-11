# ros_ws 

This repo contains the packages and nodes for task4c for eyantra.

### NOTE:
```bash
# ALWAY PULL/FETCH BEFORE PUSH
git pull

# Then push
git add .
git commit -m "PROPER MESSAGE"
git push origin main
```

## RULE:
1. Your node must satisfy eyantra dependency version
2. No venv shoul need to run any node
3. No Graph or ploting logic in code, you can use logger() for visualization.

### NOTE:

You can create ```.playground``` folder in your local in this repo to test and work on your scripts. This folder doesn't push on github.


## Message Notation
Always add a small label for your message that can categorise your message, like 'fix:' for fixing existing code, 'new:' for adding new logic, 'refactor:' for refactoring code, 'format:' for formate etc
```bash
# Message example for fixing script
git commit -m "fix: fixed the bug of collision at wall in nav.py"

# for adding new script

git commit -m "Added: New wall collison detection script added wall.py"

# Else
git commit -m "feat: Added new logic"
