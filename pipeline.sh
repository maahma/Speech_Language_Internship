#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# Setting paths to files
INPUT_TEXT_FILE="Recordings/EarningsCall_Dataset-master/Amazon.com\ Inc._20170202/Text.txt"
CLEANED_FILE="cleaned_data.txt"
stage=1
stop_stage=1


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1]; then
    log "Stage 1: Cleaning text data"
    python3 clean_text.py --input "$INPUT_FILE" --output "$CLEANED_FILE"
    echo "Data Cleaning completed. Output: "$CLEANED_FILE"  