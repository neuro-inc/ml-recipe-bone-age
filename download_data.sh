#!/bin/bash

DEST=/data
TMP=$(mktemp -d)

wget http://data.neu.ro/bone-age.zip -O $TMP/bone-age.zip
unzip $TMP/bone-age.zip -d $DEST

