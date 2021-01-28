#!/bin/bash
set -eux -o pipefail -o xtrace

GIT_USER_NAME=${GIT_USER_NAME:-"bot-${NEURO_JOB_OWNER}"}
GIT_USER_EMAIL=${GIT_USER_EMAIL:-"job--${NEURO_JOB_ID}.cluster--${NEURO_JOB_CLUSTER}@neu.ro"}
GIT_COMMIT_MSG="Auto-commit: update dataset"
GIT_TARGET_BRANCH=""

paths=""
git_branch="$GIT_TARGET_BRANCH"
git_user_name="$GIT_USER_EMAIL"
git_user_email="$GIT_USER_NAME"
git_commit_msg="$GIT_COMMIT_MSG"
for arg in "$@"
do
  key=$(echo $arg | cut -f1 -d=)
  val=$(echo $arg | cut -f2 -d=)
  case $key in
    paths) paths=$val;;
    git_branch) git_branch=$val;;
    git_user_name) git_user_name=$val;;
    git_user_email) git_user_email=$val;;
    git_commit_msg) git_commit_msg=$val;;
    *) echo "Unknown argument $key=$val: ignoring"
esac
done
test "$paths" || { echo "Empty argument paths= (allowed)"; }
test "$git_branch" || { echo "Missing required argument git_branch="; exit 1; }
test "$git_user_name" || { echo "Missing required argument git_user_name="; exit 1; }
test "$git_user_email" || { echo "Missing required argument git_user_email="; exit 1; }
test "$git_commit_msg" || { echo "Missing required argument git_commit_msg="; exit 1; }


# Script started
git config user.email git_user_email
git config user.name git_user_name
git diff | tee
for path in $(echo $paths | tr ',' ' ')
do
  dvc_path="${path}.dvc"
  echo "Git-committing dvc-file $dvc_path"
  git add $dvc_path
done
git commit --allow-empty -m "$git_commit_msg"
git push --set-upstream origin $git_branch
