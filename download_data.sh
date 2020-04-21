#!/bin/bash

DEST=${1:-/data}
TMP=$(mktemp -d)
echo "Will download dataset to $TMP and unpack it to $DEST"

wget -q http://data.neu.ro/bone-age.zip -O $TMP/bone-age.zip
# FIX: warning extra bytes at beginning or within zipfile
zip -FFv $TMP/bone-age.zip --out $TMP/bone-age-fixed.zip
unzip -q $TMP/bone-age-fixed.zip -d $DEST

