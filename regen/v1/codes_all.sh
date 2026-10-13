#!/bin/zsh
# one process per 60 views: the MPS allocator's memory grows over a long run
cd "$(dirname "$0")/../../.."
while true; do
  before=$(ls work/regen/v1/codes | wc -l)
  caffeinate -i -s work/geoenv/bin/python work/watch_mem.py 14 work/regen/v1/codes.log work/geoenv/bin/python work/regen/codes.py --limit 60
  after=$(ls work/regen/v1/codes | wc -l)
  [ "$after" -eq "$before" ] && break
done
echo ALLDONE >> work/regen/v1/codes.log
