#!/bin/bash
set -eu
if [ "$#" != 3 ]; then
    echo Usage: ./test.sh /data/train /data/test name_of_output_file
    exit 1
fi
exec bash code/_common.sh "$1" "$2" "$3" --test
