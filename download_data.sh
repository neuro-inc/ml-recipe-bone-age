#!/bin/bash

DEST=${1:-/data}
TMP=$(mktemp -d)
echo "Will download dataset to $TMP and unpack it to $DEST"

wget http://data.neu.ro/bone-age.zip -O $TMP/bone-age.zip
unzip $TMP/bone-age.zip -d $DEST

