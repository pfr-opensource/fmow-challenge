#!/bin/bash
set -eu
if [ "$#" != 1 ]; then
    echo Usage: ./train.sh /data/train
    exit 1
fi
exec bash code/_common.sh "$1" unused unused --train
