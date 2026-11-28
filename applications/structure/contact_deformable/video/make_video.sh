#!/bin/bash
# Video of the self-contact test (closing slot, Setup = SlottedBlock).
#
# The quasi-static driver writes output at the beginning and at the end of a run only. The problem
# is elastic and frictionless, so the solution does not depend on the load path: the frames are the
# final states of runs with an increasing prescribed displacement (push).
#
# usage: make_video.sh <solver executable> <work directory> <output .mp4> [n_states] [refinements]
set -e
SOLVER=$(realpath "$1")
WORK=$(realpath -m "$2")
OUTPUT=$(realpath -m "$3")
N=${4:-48}
L=${5:-3}
PUSH_MAX=0.045
HERE=$(dirname "$(realpath "$0")")
TEMPLATE=$HERE/../tests/slotted_block_self_contact.json

export DEAL_II_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$WORK/runs"
cd "$WORK"

for k in $(seq 1 "$N"); do
  kk=$(printf %03d "$k")
  push=$(python3 -c "print($PUSH_MAX * $k / $N)")
  sed -e "s/\"IsTest\": \"true\"/\"IsTest\": \"false\"/" \
      -e "s/\"RefineSpaceMin\": \"[0-9]*\"/\"RefineSpaceMin\": \"$L\"/" \
      -e "s/\"RefineSpaceMax\": \"[0-9]*\"/\"RefineSpaceMax\": \"$L\"/" \
      -e "s/\"Push\": \"[0-9.e-]*\"/\"Push\": \"$push\"/" \
      -e "s/\"VerifySearch\": \"true\"/\"VerifySearch\": \"false\"/" \
      -e "s/\"WriteOutput\": \"false\"/\"WriteOutput\": \"true\"/" \
      -e "s#\"OutputDirectory\": \"[^\"]*\"#\"OutputDirectory\": \"runs/k$kk/\"#" \
      -e "s/\"OutputName\": \"[A-Za-z_]*\"/\"OutputName\": \"slot\"/" \
      "$TEMPLATE" > "runs/k$kk.json"
done

# independent runs (results only, no timings): at most 6 at a time
ls runs/k*.json | xargs -P 6 -I{} bash -c '"$0" {} > "$(dirname {})/$(basename {} .json).log" 2>&1 || echo "FAILED {}"' "$SOLVER"

python3 -I "$HERE/render.py" "$WORK/runs" "$WORK/frames" "$PUSH_MAX" "$N"

# hold the first frame for 0.6 s and the last frame for 2.5 s, 10 frames per second
rm -rf seq && mkdir seq
i=0
add() { cp "frames/frame_$1.png" "seq/s_$(printf %04d $i).png"; i=$((i + 1)); }
for r in $(seq 1 6); do add 000; done
for k in $(seq 1 "$N"); do add "$(printf %03d "$k")"; done
for r in $(seq 1 25); do add "$(printf %03d "$N")"; done
ffmpeg -y -loglevel error -framerate 10 -i seq/s_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 16 \
  -movflags +faststart "$OUTPUT"
echo "written $OUTPUT"
