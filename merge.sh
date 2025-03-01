#!/bin/bash

# Merge main into all challenger branches

for BRANCH in $(ls .git/refs/heads);
  do git checkout $BRANCH ; 
  git merge origin/main $BRANCH ; 
done
