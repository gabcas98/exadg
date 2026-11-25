#!/bin/bash
# Runs the convergence studies for the contact report (2d Hertz + patch tests).
# usage: run_studies.sh <build directory of applications/structure/contact_rigid_obstacle>
B=${1:?build directory of the application}
cd $B
mk(){ name=$1; base=$2; shift 2; python3 - "$name" "$base" "$@" <<'PY'
import json,sys
name,base=sys.argv[1],sys.argv[2]
d=json.load(open(base))
for kv in sys.argv[3:]:
    sec,k,v=kv.split(':',2); d[sec][k]=v
d['Output']['OutputName']=name
json.dump(d,open(f'in_{name}.json','w'),indent=2)
PY
mpirun -np 12 ./solver in_$name.json > $name.log 2>&1 || echo "FAILED $name"; }

# A: indentation study, a/h ~ 16, eps = 100/delta
for spec in "3e-2 7 0.35" "1e-2 7 0.2" "3e-3 8 0.11" "1e-3 9 0.063" "3e-4 9 0.035" "1e-4 10 0.02"; do
  set -- $spec; eps=$(python3 -c "print(100/$1)")
  mk A_d$1 input_hertz.json Application:Indentation:$1 Application:LocalRefinements:$2 Application:RefinementRadius:$3 Application:PenaltyParameter:$eps
done
# B: mesh study at delta = 1e-3, eps = 1e5 (reference: L = 11)
for L in 5 6 7 8 9 10 11; do
  mk B_L$L input_hertz.json Application:Indentation:1e-3 Application:LocalRefinements:$L Application:RefinementRadius:0.063 Application:PenaltyParameter:1e5
done
# C: penalty study at delta = 1e-3, L = 9
for e in 1e2 1e3 1e4 1e5 1e6 1e7; do
  mk C_e$e input_hertz.json Application:Indentation:1e-3 Application:LocalRefinements:9 Application:RefinementRadius:0.063 Application:PenaltyParameter:$e
done
# D: patch tests
mk D_2d_q2_amg input_patch_test.json
mk D_2d_q3_mg input_patch_test.json SpatialResolution:DegreeMin:3 SpatialResolution:DegreeMax:3 Application:Preconditioner:Multigrid
mk D_2d_q2_mb input_patch_test.json Application:Preconditioner:Multigrid Application:UseMatrixBasedOperator:true
mk D_2d_q2_large input_patch_test.json Application:Pressure:0.15 Application:Preconditioner:Multigrid
mk D_3d_q2_mg input_patch_test.json General:Dim:3 SpatialResolution:RefineSpaceMin:1 SpatialResolution:RefineSpaceMax:1 Application:Preconditioner:Multigrid
mk D_3d_q3_large input_patch_test.json General:Dim:3 SpatialResolution:DegreeMin:3 SpatialResolution:DegreeMax:3 SpatialResolution:RefineSpaceMin:1 SpatialResolution:RefineSpaceMax:1 Application:Preconditioner:Multigrid Application:Pressure:0.15
echo ALL_DONE
