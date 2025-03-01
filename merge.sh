#!/bin/bash

# Merge main into all challenger branches

if [[ `git branch | grep '*' | grep main | wc -l` -eq "0" ]]; then
    echo "Error: Please checkout the main branch"
    exit 1
fi

for BRANCH in $(ls .git/refs/heads);
  do git checkout $BRANCH ; 
  git merge origin/main $BRANCH ; 
done

git checkout main
