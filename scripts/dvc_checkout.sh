#!/bin/bash
set -eux -o pipefail -o xtrace

CACHE_NAME=mounted_cache

dvc_cache_path=""
for arg in "$@"
do
  key=$(echo $arg | cut -f1 -d=)
  val=$(echo $arg | cut -f2 -d=)
  case $key in
    dvc_cache_path) dvc_cache_path=$val;;
    *) echo "Unknown argument $key=$val: ignoring"
esac
done
test "$dvc_cache_path" || { echo "Missing required argument dvc_cache_path="; exit 1; }


# Script started
dvc init -q -f
dvc remote add -d $CACHE_NAME $dvc_cache_path
dvc pull -f
