#!/bin/bash
# Exhaustive vs. Verlet search on the TwoBlocks setup: same mesh, loads and tolerances, only the
# search strategy differs. Runs are executed one after the other (serial, one MPI process) so that
# the wall times are not affected by concurrent runs.
#
# usage: run_benchmark.sh <solver executable> <output directory> [refinement levels] [surfaces]
set -e
# one thread per run (deal.II/TBB and OpenMP), see the comment above
export DEAL_II_NUM_THREADS=1 OMP_NUM_THREADS=1
SOLVER=$(realpath "$1")
OUT=$2
LEVELS=${3:-"3 4 5 6"}
SURFACES=${4:-"Faces Boundary"}
INPUT=$(dirname "$(realpath "$0")")/input.json
mkdir -p "$OUT"
cd "$OUT"
for surfaces in $SURFACES; do
  for L in $LEVELS; do
    for search in Exhaustive Verlet; do
      name=${surfaces}_L${L}_${search}
      sed -e "s/\"RefineSpaceMin\": \"[0-9]*\"/\"RefineSpaceMin\": \"$L\"/" \
          -e "s/\"RefineSpaceMax\": \"[0-9]*\"/\"RefineSpaceMax\": \"$L\"/" \
          -e "s/\"ContactSurfaces\": \"[A-Za-z]*\"/\"ContactSurfaces\": \"$surfaces\"/" \
          -e "s/\"Search\": \"[A-Za-z]*\"/\"Search\": \"$search\"/" \
          -e "s/\"OutputName\": \"[A-Za-z_]*\"/\"OutputName\": \"$name\"/" \
          "$INPUT" > "$name.json"
      echo "running $name"
      "$SOLVER" "$name.json" > "$name.log" 2>&1
    done
  done
done
