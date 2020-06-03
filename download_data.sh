#!/bin/bash

DATASET=${1:-bone-age-full.zip}
DEST=${2:-/data}
TMP=$(mktemp -d)
echo "Will download dataset to $TMP and unpack it to $DEST"

wget http://data.neu.ro/$DATASET -O $TMP/out.zip
unzip $TMP/out.zip -d $DEST
