#!/bin/bash
# Runs the convergence studies for the contact report (2d Hertz + patch tests).
# usage: run_studies.sh <build directory of applications/structure/contact_rigid_obstacle>
B=${1:?build directory of the application}
# input files and comparison scripts of the application
SRC=$(dirname "$(readlink -f "$0")")/../../applications/structure/contact_rigid_obstacle
cp "$SRC"/input_*.json "$SRC"/*_comparison.py "$B"/
cd $B
# Settings of the earlier studies (A-H): plain damped Newton method and Chebyshev degree 5, which
# were the defaults of the application when these studies were run. They are prepended, so that
# options given to mk override them.
EXTRA=${EXTRA-"Application:NonmonotoneSteps:0 Application:LineSearchRefinements:0 Application:MaxIterDamping:10 Application:SmootherIterations:5"}
mk(){ name=$1; base=$2; shift 2; python3 - "$name" "$base" $EXTRA "$@" <<'PY'
import json,sys
name,base=sys.argv[1],sys.argv[2]
d=json.load(open(base))
for kv in sys.argv[3:]:
    sec,k,v=kv.split(':',2); d[sec][k]=v
d['Output']['OutputName']=name
json.dump(d,open(f'in_{name}.json','w'),indent=2)
PY
mpirun -np 8 ./solver in_$name.json > $name.log 2>&1 || echo "FAILED $name"; }

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

# E: Nitsche mesh study at delta = 1e-3, gamma_0 E = 10 (reference: L = 11)
for L in 5 6 7 8 9 10 11; do
  mk E_L$L input_hertz.json Application:ContactMethod:Nitsche Application:Indentation:1e-3 Application:LocalRefinements:$L Application:RefinementRadius:0.063 Application:PenaltyParameter:10
done
# F: Nitsche parameter study at delta = 1e-3, L = 9
for g in 0.3 1 3 10 30 100 300; do
  mk F_g$g input_hertz.json Application:ContactMethod:Nitsche Application:Indentation:1e-3 Application:LocalRefinements:9 Application:RefinementRadius:0.063 Application:PenaltyParameter:$g
done
# G: Nitsche patch tests
mk G_2d_q2_amg input_patch_test.json Application:ContactMethod:Nitsche Application:PenaltyParameter:10
mk G_2d_q3_mg input_patch_test.json Application:ContactMethod:Nitsche Application:PenaltyParameter:10 SpatialResolution:DegreeMin:3 SpatialResolution:DegreeMax:3 Application:Preconditioner:Multigrid
mk G_3d_q3_large input_patch_test.json Application:ContactMethod:Nitsche Application:PenaltyParameter:10 General:Dim:3 SpatialResolution:DegreeMin:3 SpatialResolution:DegreeMax:3 SpatialResolution:RefineSpaceMin:1 SpatialResolution:RefineSpaceMax:1 Application:Preconditioner:Multigrid Application:Pressure:0.15
# H: 3d Hertz (sphere), multigrid study. The runs h3d_d1e-3_L6 and N_h3d_L6 of the report ("before
# fix") were computed before passing the time to the multigrid levels and cannot be reproduced with
# the current code; with it, they coincide with fix_h3d_d1e-3_L6 and fix_N_h3d_L6.
P3="Application:Indentation:1e-3 Application:RefinementRadius:0.05 Application:Preconditioner:Multigrid General:Dim:3 Application:ContactMethod:Penalty"
N3="Application:Indentation:1e-3 Application:RefinementRadius:0.05 Application:Preconditioner:Multigrid General:Dim:3 Application:ContactMethod:Nitsche"
mk fix_h3d_d1e-3_L6 input_hertz.json $P3 Application:LocalRefinements:6 Application:PenaltyParameter:1e5 Application:SmootherIterations:5
mk fix_N_h3d_L6 input_hertz.json $N3 Application:LocalRefinements:6 Application:PenaltyParameter:10 Application:SmootherIterations:5
mk amg_Nitsche input_hertz.json $N3 Application:LocalRefinements:6 Application:PenaltyParameter:10 Application:Preconditioner:AMG
mk m_pen_e1e4 input_hertz.json $P3 Application:LocalRefinements:6 Application:PenaltyParameter:1e4 Application:SmootherIterations:5
mk m_nit_g100 input_hertz.json $N3 Application:LocalRefinements:6 Application:PenaltyParameter:100 Application:SmootherIterations:5
for k in 5 8; do
  mk z_pen_e1e6_pj$k input_hertz.json $P3 Application:LocalRefinements:6 Application:PenaltyParameter:1e6 Application:SmootherIterations:$k
done
mk z_pen_e1e5_pj8 input_hertz.json $P3 Application:LocalRefinements:6 Application:PenaltyParameter:1e5 Application:SmootherIterations:8
mk z_nit_g10_pj8 input_hertz.json $N3 Application:LocalRefinements:6 Application:PenaltyParameter:10 Application:SmootherIterations:8
mk z_nit_g100_pj8 input_hertz.json $N3 Application:LocalRefinements:6 Application:PenaltyParameter:100 Application:SmootherIterations:8
for k in 5 8 12; do
  mk y_pen_pj$k input_hertz.json $P3 Application:LocalRefinements:7 Application:PenaltyParameter:1e5 Application:SmootherIterations:$k
  mk y_nit_pj$k input_hertz.json $N3 Application:LocalRefinements:7 Application:PenaltyParameter:10 Application:SmootherIterations:$k
done
# I: Newton's method for stiff contact: plain damped Newton (K0), non-monotone full steps (K1),
# and the default of the application (non-monotone with K = 2, factor 1e6, 20 damping steps and 6
# bisection steps of the line search)
K0="Application:NonmonotoneSteps:0 Application:LineSearchRefinements:0 Application:MaxIterDamping:10"
K1="Application:NonmonotoneSteps:1 Application:NonmonotoneFactor:1e2 Application:LineSearchRefinements:0 Application:MaxIterDamping:10"
V3="Application:NonmonotoneSteps:3 Application:NonmonotoneFactor:1e6 Application:MaxIterDamping:20 Application:LineSearchRefinements:6"
H2="Application:ContactMethod:Nitsche Application:Indentation:1e-3 Application:RefinementRadius:0.063"
for v in K0 K1 V3; do
  opts=${!v}
  mk nw_F_g10_$v input_hertz.json $H2 Application:LocalRefinements:9 Application:PenaltyParameter:10 $opts
  mk nw_F_g100_$v input_hertz.json $H2 Application:LocalRefinements:9 Application:PenaltyParameter:100 $opts
  mk nw_F_g300_$v input_hertz.json $H2 Application:LocalRefinements:9 Application:PenaltyParameter:300 $opts
  mk nw_N_h_g1000_$v input_hertz.json $H2 Application:LocalRefinements:9 Application:PenaltyParameter:1000 $opts
  mk nw_E_L10_$v input_hertz.json $H2 Application:LocalRefinements:10 Application:PenaltyParameter:10 $opts
  mk nw_E_L11_$v input_hertz.json $H2 Application:LocalRefinements:11 Application:PenaltyParameter:10 $opts
  mk nw_C_e1e5_$v input_hertz.json Application:ContactMethod:Penalty Application:Indentation:1e-3 Application:RefinementRadius:0.063 Application:LocalRefinements:9 Application:PenaltyParameter:1e5 $opts
  mk nw_C_e1e7_$v input_hertz.json Application:ContactMethod:Penalty Application:Indentation:1e-3 Application:RefinementRadius:0.063 Application:LocalRefinements:9 Application:PenaltyParameter:1e7 $opts
  mk nw_y_nit_pj8_$v input_hertz.json $N3 Application:LocalRefinements:7 Application:PenaltyParameter:10 Application:SmootherIterations:8 $opts
  mk nw_y_pen_pj8_$v input_hertz.json $P3 Application:LocalRefinements:7 Application:PenaltyParameter:1e5 Application:SmootherIterations:8 $opts
done
# T: Nitsche variants theta = 1, 0, -1 (robust Newton method)
for th in 1 0 -1; do
  for g in 0.001 0.01 0.1 1 10 100; do
    mk T_th${th}_g$g input_hertz.json $V3 Application:ContactMethod:Nitsche Application:Indentation:1e-3 Application:LocalRefinements:9 Application:RefinementRadius:0.063 Application:NitscheTheta:$th Application:PenaltyParameter:$g
  done
done
# reference of the theta study
mk v3_F_g10 input_hertz.json $V3 Application:ContactMethod:Nitsche Application:Indentation:1e-3 Application:LocalRefinements:9 Application:RefinementRadius:0.063 Application:PenaltyParameter:10

# J: frictional contact (robust Newton method is the default of the application)
EXTRA=""
mk sliding_block input_sliding_block.json
mk sb2d_mg input_sliding_block.json Application:Preconditioner:Multigrid
mk sb3d_mg input_sliding_block.json General:Dim:3 SpatialResolution:RefineSpaceMin:2 SpatialResolution:RefineSpaceMax:2 Application:Preconditioner:Multigrid
for s in 5e-5 1e-4 2e-4 4e-4; do
  mk cat_s$s input_cattaneo.json Application:TangentialShift:$s
done
echo ALL_DONE
