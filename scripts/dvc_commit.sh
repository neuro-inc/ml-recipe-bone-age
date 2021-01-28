#!/bin/bash
set -eux -o pipefail -o xtrace

paths=""
for arg in "$@"
do
  key=$(echo $arg | cut -f1 -d=)
  val=$(echo $arg | cut -f2 -d=)
  case $key in
    paths) paths=$val;;
    *) echo "Unknown argument $key=$val: ignoring"
esac
done
test "$paths" || { echo "Missing required argument paths="; exit 1; }


# Script started
echo "Commiting changes for paths: $paths"
dvc status
for path in $(echo $paths | tr ',' ' ')
do
  dvc_path="${path}.dvc"
  echo "Dvc-committing path $path -> $dvc_path"
  if [[ -e "$dvc_path" ]]
  then
    dvc commit --force $dvc_path
  else
    dvc add $path --file $dvc_path
  fi
done

echo "Pushing changes to remote dvc cache"
dvc push
