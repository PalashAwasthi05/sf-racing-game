#!/bin/zsh
# LiDAR skyline alignment of every view of an area's sides, in chunks (the segmenter's GPU memory grows), resuming.
# Usage: work/areas/align_area.sh <area name>
cd "$(dirname "$0")/.."
A=areas/$1; R=$A/regen; LOG=$A/align.log
export SFW_AREA=$A REGEN_ROOT=$R PYTHONUNBUFFERED=1
py=geoenv/bin/python
for k in $(seq 1 200); do
  b=$($py -c "import json,os;f='$R/alignment.json';print(len(json.load(open(f))) if os.path.exists(f) else 0)")
  caffeinate -i -s $py watch_mem.py 12 $LOG $py regen/align.py --root $R --limit 120
  n=$($py -c "import json,os;f='$R/alignment.json';print(len(json.load(open(f))) if os.path.exists(f) else 0)")
  echo "aligned $b -> $n"; [ "$n" = "$b" ] && break
done
echo ALIGN-DONE $1
