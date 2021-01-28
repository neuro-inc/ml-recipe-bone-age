#!/bin/bash
set -eux -o pipefail -o xtrace

GIT_REPO=${GIT_REPO:-"git@github.com:neuro-inc/ml-recipe-bone-age.git"}

git_branch=""
create_git_branch=""
project_path="."
for arg in "$@"
do
  key=$(echo $arg | cut -f1 -d=)
  val=$(echo $arg | cut -f2 -d=)
  case $key in
    git_branch) git_branch=$val;;
    create_git_branch) create_git_branch=$val;;
    project_path) project_path=$val;;
    *) echo "Unknown argument $key=$val: ignoring"
esac
done
test "$git_branch" || { echo "Missing required argument git_branch="; exit 1; }
test "$create_git_branch" || echo "Empty argument create_git_branch= (allowed)"
test "$project_path" || { echo "Missing required argument project_path="; exit 1; }


# Script started
mkdir -p ~/.ssh
echo "IdentityFile ~/.ssh/id-rsa" > ~/.ssh/config
ssh-keyscan github.com >> ~/.ssh/known_hosts
git clone $GIT_REPO $project_path
cd $project_path
git checkout $git_branch

if [[ "$create_git_branch" ]] && [[ "$create_git_branch" != "$git_branch" ]]
then
  git checkout -b $create_git_branch

  origin_create_git_branch=origin/$create_git_branch
  git show-branch remotes/$origin_create_git_branch \
    && { echo "error: remote branch $origin_create_git_branch alrady exists"; exit 1 ;} \
    || echo "ok, remote branch $origin_create_git_branch does not exists"
fi