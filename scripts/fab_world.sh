#!/bin/bash
# FAB world motion + scene from one video in one command (world-mode entry point, estimated intrinsics). The steps of
# --profile paper (the paper's world pipeline; the default takes a fast path on fixed cameras: --profile below):
#   frames -> ViPE 1.2 + UniDepth (--depth) -> dense depth -> GeoCalib -> the focal rule (bench/scene_recon/focal_rule.py:
#   a static camera, p90 camera-centre spread < 0.5 % of the median static depth: median(ViPE, GeoCalib, MoGe-2) focal,
#   ViPE re-run with K fixed) -> person segmentation -> FAB (FAB-L, or --fab S: FAB-S; the released profile and SMPL
#   fitter, the released detector) -> contacts (the model's image contact head -> the causal video contact model, world
#   frame) -> world reconstruction (ViPE depth) -> MVS (static cameras: its no-parallax rule keeps the ViPE depth) -> world
#   reconstruction (MVS depth) -> joint human-scene step
# -> <out>/<name>.npz: the world body (pose / betas / trans / joints, z-up metric world, gravity -z), cameras (T_world_cam,
#    K), contacts, scale_fix, notes; <out>/<name>_scene.ply + <name>_hfield.npz (the scene); intermediate steps in <out>/work.
# Needs the world-mode tools (docs/world.md): the ViPE envs, COLMAP and MoGe-2 are installed here when missing.
#   bash scripts/fab_world.sh <video | frame dir> [--out DIR] [--name NAME] [--fps auto | F] [--fab L|S] [--subject_boxes box.npz] [--focal F]
#        [--bundle model.pt] [--fitter model.pt] [--contact_head model.pt] [--contact_video model.pt] [--mvs_reuse DIR]
#        [--profile default|paper] [--camera vipe|auto|static] [--focal_rule geocalib|median|nc3] [--floor_up 1|0] [--vipe_mpix N]
#        [--vipe_depth MODEL] [--vipe_stride S] [--vipe_chunk_real 0|1] [--vipe_max_side PX] [--vipe_no_instance 0|1]
#        [--single_pass 0|1] [--mvs 1|0]
#        [--track iou|oc] [--track_kw JSON] [--det_size 640] [--init_box x0,y0,x1,y1 | box.npz] [--init_frame 0]
#        [--motion off|learned] [--motion_model world_motion.pt] [--motion_fps auto|resample] [--refine none|physics]
#   (--subject_boxes: npz box (T, 4) xyxy of the target person per frame, NaN unknown: identity only; default the most
#    prominent tracked person. --focal F: opt-in for a calibrated camera, the known focal in px: one fixed-K ViPE run, no
#    focal rule; no paper number uses it. GPU: the first of CUDA_VISIBLE_DEVICES, so several runs can share a machine.
#    --bundle / --fitter / --contact_head / --contact_video: another body model, SMPL fitter or contact pair (e.g. one
#    you trained); defaults = the released ones under $FAB_WEIGHTS: fab-l (S: fab-s), shared/smpl_fitter.pt, the body
#    model itself as the image contact head, shared/contact_video.pt; all recorded in the export's notes. --mvs_reuse
#    DIR: an existing COLMAP PatchMatch workspace root (DIR/<name>[/<name>]/dense/stereo/depth_maps) of the same cameras:
#    linked (never written) and only mvs_depth's filtering re-run with this run's body (mvs_depth --reuse 1); a clip whose
#    workspace is incomplete gets a fresh PatchMatch run in a scratch dir. --profile sets the defaults of --camera /
#    --single_pass / --vipe_depth / --vipe_mpix / --mvs / --vipe_stride / --vipe_chunk_real / --vipe_max_side / --fps
#    and the frame rate the body smoothing is defined at (each flag given overrides it): default (no flag needed) = auto
#    / 1 / adaptive_unidepth-l / 2000 / 0 / 2 / 1 / 960 / auto, smoothing at 30 fps scaled to the clip's rate: the
#    static-camera fast path, and on a moving camera ViPE on every 2nd frame at a 960 px long side, in chunks of the
#    same real duration, without MVS; paper = vipe / 0 / ViPE's / off / 1 / 1 / 0 / 0 (full frames) / 30, smoothing in
#    the clip's own frames: the pipeline of the paper's world rows. A slower moving-camera run: --vipe_stride 1 (ViPE on
#    every frame), --vipe_max_side 0 (full frames; both the same accuracy on the development sets) and / or
#    --mvs 1 (COLMAP multi-view stereo, several times slower). The no-VDA depth and --mvs 0 belong together: on ViPE's
#    default (VDA) depth, dropping MVS cost more. --camera: vipe = the ViPE path above;
#    auto = a fixed camera (bench/world_static.py check: the frames do not move) takes the static-camera fast path, any
#    other the ViPE path; static = the fast path without the check (the camera is known to be fixed). The fast path: no
#    ViPE run, identity cameras, focal from GeoCalib / MoGe-2 (--focal_rule; --focal: the given one; nc3: the median of
#    GeoCalib, Pi4 and VGGT-Omega, non-commercial models, via FAB_NC_PY / FAB_PI4_DIR / FAB_PI4_WEIGHTS / FAB_VGGT_DIR /
#    FAB_VGGT_CKPT, bench/scene_recon/focal_nc.py), MoGe-2 metric depth
#    + Mask2Former dynamic masks (bench/world_static.py depth), GeoCalib gravity with that focal, refined by the floor
#    under the person (--floor_up 1, bench/world_static.py gravity; 0: GeoCalib's). --vipe_mpix N: ViPE runs
#    in overlapping chunks when frames x megapixels > N (bench/world_vipe.py --chunk_mpix; the depth step merges them), so
#    long videos fit the GPU. --vipe_depth MODEL: ViPE's dense depth model (its depth_align_model; default ViPE's
#    adaptive_unidepth-l_svda; adaptive_unidepth-l skips the Video Depth Anything alignment: the same cameras, ~3x less GPU
#    memory on EMDB). --vipe_stride S: ViPE on every S-th frame (+ the last), poses interpolated to every frame, depth
#    on those frames only (bench/world_vipe.py --stride; 1: every frame); --vipe_chunk_real 1: the --vipe_mpix chunks
#    counted in real frames, so a chunk spans the same time at any stride (0: in ViPE's frames). --vipe_max_side PX: ViPE
#    on frames resized to a long side <= PX (0: full frames; intrinsics back to the full frames); --vipe_no_instance 1:
#    ViPE without its GroundingDINO / SAM instance masks (people masked downstream by the Mask2Former seg only). --mvs 0: no COLMAP
#    multi-view stereo on a moving camera (the TSDF of ViPE's depth alone, as a static
#    camera always gets). --single_pass 1: the contact model's image features come from the FAB stage's own body pass
#    (recon fab --contact_feats 1, scripts/contact_clips.py tracks) instead of a second FAB-L pass over every frame.
#    --track oc: opt-in subject tracking (the video mode with persistent Kalman / OC-SORT ids and the detector on every
#    frame, the subject chosen over the whole clip by FAB's keypoint extent x centrality, never another person;
#    bench/sonic_motion.picks_video), --track_kw its options as
#    JSON (bench/sonic_motion.SUBJECT_KW); default iou: the released subject pick. --det_size: the detector's input size,
#    default 640; 960 finds smaller / farther people. --init_box: one prompt box for the subject (implies --track oc;
#    xyxy px on frame --init_frame, or an npz box (T, 4) whose first finite row is used): the tracked person it overlaps
#    is followed through the clip, else the automatic choice. --motion learned (opt-in): FAB-M (fab/world_motion.py)
#    re-places the exported body from the per-frame bodies, cameras and contacts; the kinematic export kept in work/kin/;
#    static cameras only: a moving camera's export passes through unchanged; 30 fps as is, other rates resampled to 30
#    and back from 24 fps up, slower clips passed through; --motion_fps resample: resampled from 10 fps up; --motion off
#    (default): the kinematic export. --refine physics: the optional physical-plausibility post-process
#    bench/scene_recon/world_refine.py on the export (footprint ground plane + levelling, per-frame depth along the camera
#    rays, planted feet locked on the ground); the unrefined body stays in the export as pose_in / trans_in; default none)
# Tools (override with env vars): FAB_HOME (~/.cache/fab), FAB_ENVS ($FAB_HOME/envs), FAB_VIPE_PY ($FAB_ENVS/vipe/bin/python),
#   FAB_VIPE12_PY ($FAB_ENVS/vipe12/bin/python), VIPE12_DIR ($FAB_ENVS/src/vipe12), FAB_COLMAP ($FAB_ENVS/colmap/bin/colmap),
#   FAB_LOAD_WORKERS (4: threads decoding the frames ahead of the FAB and contact stages; 1 = serial), FAB_FAB_FAST (0; 1 =
#   FABPredictor(fast=True) in the FAB stage: the TensorRT body engine when fab-build-trt built one, else compiled PyTorch).
set -uo pipefail
IN=""; OUT=output/fab_world; NAME=""; FPS=-; FOCAL=""; SB=""; FAB=L; BUNDLE=""; FITTER=""; CHEAD=""; CVID=""; MVSR=""; FRULE=geocalib; FLUP=1
PROFILE=default; CAM=-; SPASS=-; VDEPTH=-; VMPIX=-; MVS=-; VSTRIDE=-; VCREAL=-; VMAXS=-; VNOINST=0  # "-": from --profile
TRACK=iou; TRACK_KW=""; DET_SIZE=640; INIT_BOX=""; INIT_FRAME=0
MOTION=off; MMODEL=""; MFPS=auto; REFINE=none
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2;; --name) NAME=$2; shift 2;; --fps) FPS=$2; shift 2;; --focal) FOCAL=$2; shift 2;;
    --subject_boxes) SB=$2; shift 2;; --fab) FAB=$2; shift 2;; --bundle) BUNDLE=$2; shift 2;; --fitter) FITTER=$2; shift 2;;
    --contact_head) CHEAD=$2; shift 2;; --contact_video) CVID=$2; shift 2;; --mvs_reuse) MVSR=$2; shift 2;;
    --profile) PROFILE=$2; shift 2;; --camera) CAM=$2; shift 2;; --focal_rule) FRULE=$2; shift 2;; --vipe_mpix) VMPIX=$2; shift 2;; --vipe_stride) VSTRIDE=$2; shift 2;; --vipe_chunk_real) VCREAL=$2; shift 2;; --vipe_max_side) VMAXS=$2; shift 2;; --vipe_no_instance) VNOINST=$2; shift 2;; --floor_up) FLUP=$2; shift 2;; --single_pass) SPASS=$2; shift 2;; --vipe_depth) VDEPTH=$2; shift 2;; --mvs) MVS=$2; shift 2;;
    --track) TRACK=$2; shift 2;; --track_kw) TRACK_KW=$2; shift 2;; --det_size) DET_SIZE=$2; shift 2;;
    --init_box) INIT_BOX=$2; shift 2;; --init_frame) INIT_FRAME=$2; shift 2;;
    --motion) MOTION=$2; shift 2;; --motion_model) MMODEL=$2; shift 2;; --motion_fps) MFPS=$2; shift 2;; --refine) REFINE=$2; shift 2;;
    -h|--help) sed -n 2,72p $0; exit 0;; *) IN=$1; shift;;
  esac
done
[ -n "$IN" ] || { sed -n 2,72p $0; exit 1; }
case $PROFILE in  # the defaults of the options left unset: camera / single_pass / vipe_depth / vipe_mpix / mvs / vipe_stride /
  # vipe_chunk_real / fps; the frame rate the body smoothing's sigmas are given at (clip: the clip's own frames); vipe_max_side
  default) P=(auto 1 adaptive_unidepth-l 2000 0 2 1 auto 30 960);;
  paper) P=(vipe 0 "" "" 1 1 0 30 clip 0);;
  *) echo "--profile default|paper"; exit 1;;
esac
[ "$CAM" = - ] && CAM=${P[0]}; [ "$SPASS" = - ] && SPASS=${P[1]}; [ "$VDEPTH" = - ] && VDEPTH=${P[2]}
[ "$VMPIX" = - ] && VMPIX=${P[3]}; [ "$MVS" = - ] && MVS=${P[4]}; [ "$VSTRIDE" = - ] && VSTRIDE=${P[5]}; [ "$VCREAL" = - ] && VCREAL=${P[6]}
[ "$FPS" = - ] && FPS=${P[7]}; SREF=${P[8]}; [ "$VMAXS" = - ] && VMAXS=${P[9]}
case $CAM in vipe|auto|static) ;; *) echo "--camera vipe|auto|static"; exit 1;; esac
[ -n "$SB" ] && [ ! -f "$SB" ] && { echo "--subject_boxes $SB: not found"; exit 1; }
[ -n "$INIT_BOX" ] && [ $TRACK != oc ] && { echo "[fab_world] --init_box: --track oc"; TRACK=oc; }
[[ "$INIT_BOX" == *.npz ]] && INIT_BOX=$(realpath "$INIT_BOX")
case $TRACK in iou|oc) ;; *) echo "--track iou|oc"; exit 1;; esac
case $MOTION in off|learned) ;; *) echo "--motion off|learned"; exit 1;; esac
case $MFPS in auto|resample) ;; *) echo "--motion_fps auto|resample"; exit 1;; esac
case $REFINE in none|physics) ;; *) echo "--refine none|physics"; exit 1;; esac
IN=$(realpath "$IN"); [ -n "$SB" ] && SB=$(realpath "$SB"); mkdir -p $OUT; OUT=$(cd $OUT && pwd)
for v in BUNDLE FITTER CHEAD CVID MVSR MMODEL; do [ -n "${!v}" ] && { [ -e "${!v}" ] || { echo "$v ${!v}: not found"; exit 1; }; printf -v $v %s "$(realpath "${!v}")"; }; done
R=$(cd $(dirname $0)/.. && pwd); cd $R
FAB_HOME=${FAB_HOME:-$HOME/.cache/fab}; FAB_ENVS=${FAB_ENVS:-$FAB_HOME/envs}; FAB_WEIGHTS=${FAB_WEIGHTS:-$FAB_HOME/weights}
export FAB_HOME FAB_ENVS FAB_WEIGHTS
case $FAB in L) CK=$FAB_WEIGHTS/fab-l/model.pt;; S) CK=$FAB_WEIGHTS/fab-s/model.pt;; *) echo "--fab L|S"; exit 1;; esac  # FAB-L / FAB-S
CK=${BUNDLE:-$CK}; FITTER=${FITTER:-$FAB_WEIGHTS/shared/smpl_fitter.pt}
for f in $CK $FITTER; do [ -f $f ] || { echo "$f: not found (download the FAB weights: README.md)"; exit 1; }; done
GPU=${CUDA_VISIBLE_DEVICES:-0}; GPU=${GPU%%,*}  # ViPE's --gpu: this run's GPU (world_vipe sets CUDA_VISIBLE_DEVICES from it)
V12=${VIPE12_DIR:-$FAB_ENVS/src/vipe12}; PY12=${FAB_VIPE12_PY:-$FAB_ENVS/vipe12/bin/python}; PV=${FAB_VIPE_PY:-$FAB_ENVS/vipe/bin/python}
export FAB_VIPE12_PY=$PY12 FAB_VIPE_PY=$PV VIPE12_DIR=$V12
COLMAP=${FAB_COLMAP:-$FAB_ENVS/colmap/bin/colmap}; export FAB_COLMAP=$COLMAP
HEAD=${CHEAD:-$CK}; VID=${CVID:-$FAB_WEIGHTS/shared/contact_video.pt}  # the image contact head is part of the body model
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
c=$(basename "$IN"); [ -d "$IN" ] || c=${c%.*}; c=${NAME:-$c}; G=$OUT/work; L=$G/clips.json; Lc=$L
mkdir -p $G/logs $G/focal $G/ours $G/ours_v $G/ours_m $G/scratch
ln -sfn $G/ours/fab $G/ours_v/fab; ln -sfn $G/ours/fab $G/ours_m/fab
log() { echo "== $1 $(date +%H:%M)"; }
bash bench/baselines/setup_vipe12.sh > $G/logs/setup_vipe12.log 2>&1 || { tail -n 20 $G/logs/setup_vipe12.log; exit 1; }
bash bench/baselines/setup_vipe.sh > $G/logs/setup_vipe.log 2>&1 || tail -n 5 $G/logs/setup_vipe.log
$PV -c "import imageio.v2" 2>/dev/null || uv pip install --python $PV imageio imageio-ffmpeg > /dev/null 2>&1
[ -x "$(command -v $COLMAP)" ] || bash scripts/colmap_setup.sh > $G/logs/colmap_setup.log 2>&1
bash bench/videomimic/setup_recon.sh > $G/logs/setup_recon.log 2>&1 || tail -n 5 $G/logs/setup_recon.log
if [ -z "$FOCAL" ] && ! python -c "import moge" 2>/dev/null; then
  python -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > $G/torch_pin.txt
  # MoGe at the commit the paper's runs used (MoGe-2 weights from Hugging Face at run time)
  python -m pip install -q -c $G/torch_pin.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" > /dev/null 2>&1
fi
# the clip: a frame directory as it is (sorted images), a video decoded to JPEG frames
python - "$IN" $c $FPS $L $G/frames <<'PY' || exit 1
import json, os, sys, cv2
from fab.clip_fps import clip_fps
src, c, arg, L, fd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
fps, how = clip_fps(src, arg)  # auto: the video's metadata rate, snapped to 30 within 1 %; a frame directory 30 unless --fps
print(f"[fab_world] {c}: fps {fps:g} ({how})")
if os.path.isdir(src):
    imgs = [os.path.abspath(os.path.join(src, f)) for f in sorted(os.listdir(src)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
else:
    os.makedirs(fd, exist_ok=True); cap, imgs = cv2.VideoCapture(src), []
    while True:
        ok, f = cap.read()
        if not ok: break
        imgs.append(f"{fd}/{len(imgs):05d}.jpg"); cv2.imwrite(imgs[-1], f, [cv2.IMWRITE_JPEG_QUALITY, 95])
json.dump({c: {"images": imgs, "fps": fps, "source": os.path.abspath(src)}}, open(L, "w"), indent=0)
print(c, len(imgs), "frames")
PY
[ -n "$SB" ] && { mkdir -p $G/subject_boxes; cp $SB $G/subject_boxes/$c.npz; }
log "setup ($c)"

# A: cameras + depth, GeoCalib, the focal rule (bench/scene_recon/focal_rule.py)
FR="python -m bench.scene_recon.focal_rule --work $G --clip $c"
ST=10  # 0: the static-camera fast path (a fixed camera, no ViPE run)
if [ $CAM = static ]; then ST=0; elif [ $CAM = auto ]; then
  python -m bench.world_static check --list $Lc --clip $c --out $G/static > $G/logs/static_check.log 2>&1; ST=$?
  [ $ST = 0 ] || [ $ST = 10 ] || { echo "camera check failed"; tail -n 5 $G/logs/static_check.log; exit 1; }
  cat $G/logs/static_check.log
fi
if [ $ST = 0 ]; then  # fixed camera: focal (--focal, else the rule on GeoCalib / MoGe-2), MoGe-2 depth, GeoCalib gravity
  if [ -n "$FOCAL" ]; then $FR given --focal $FOCAL; else
    $PV -m bench.scene_recon.focal_compare geocalib --list $Lc --out $G/focal/geo_$c.json > $G/logs/focal_geo.log 2>&1
    [ $FRULE = median ] && python -m bench.scene_recon.focal_compare moge --list $Lc --out $G/focal/moge_$c.json > $G/logs/focal_moge.log 2>&1
    if [ $FRULE = nc3 ]; then
      for m in pi4 vggt; do ${FAB_NC_PY:?FAB_NC_PY: a python with Pi4 / VGGT-Omega imports} -m bench.scene_recon.focal_nc $m --list $Lc --out $G/focal/${m}_$c.json > $G/logs/focal_$m.log 2>&1; done
    fi
    $FR fixed --rule $FRULE || { echo "focal rule failed"; exit 1; }
  fi
  [ -f $G/static/dense/$c.npz ] || python -m bench.world_static depth --list $Lc --clip $c --out $G/static --work $G > $G/logs/static_depth.log 2>&1
  [ -f $G/geocalib_st/$c.npz ] || $PV -m bench.world_geocalib --list $Lc --cams $G/static/cams --out $G/geocalib_st --max_frames 24 > $G/logs/geo_st.log 2>&1
  [ -f $G/static/dense/$c.npz ] && [ -f $G/geocalib_st/$c.npz ] || { echo "static camera path failed"; tail -n 5 $G/logs/static_depth.log $G/logs/geo_st.log; exit 1; }
  [ $FLUP = 1 ] && python -m bench.world_static gravity --list $Lc --clip $c --out $G/static --work $G > $G/logs/static_gravity.log 2>&1 && cat $G/logs/static_gravity.log
  grep -h "\[static\]" $G/logs/static_depth.log | cut -c1-200
elif [ -n "$FOCAL" ]; then  # opt-in calibrated camera: one fixed-K ViPE run (the pipeline's pinhole keeps the principal point centred)
  $FR given --focal $FOCAL
else
  [ -f $G/vipe/cams/$c.npz ] || $PV -m bench.world_vipe --list $Lc --out $G/vipe --python $PY12 --vipe $V12 --depth --gpu $GPU --work $G/scratch/vipe_$c ${VMPIX:+--chunk_mpix $VMPIX} ${VDEPTH:+--depth_align $VDEPTH} --stride $VSTRIDE --chunk_real $VCREAL --max_side $VMAXS --no_instance $VNOINST > $G/logs/vipe.log 2>&1
  [ -f $G/vipe/dense/$c.npz ] || $PY12 -m bench.videomimic.vipe_depth --vipe_out $G/vipe --vipe $V12 --name $c > $G/logs/dense.log 2>&1
  [ -f $G/geocalib/$c.npz ] || $PV -m bench.world_geocalib --list $Lc --cams $G/vipe/cams --out $G/geocalib > $G/logs/geo.log 2>&1
  [ -f $G/vipe/cams/$c.npz ] && [ -f $G/vipe/dense/$c.npz ] || { echo "ViPE failed"; tail -n 5 $G/logs/vipe.log $G/logs/dense.log; exit 1; }
  $FR static; rc=$?
  [ $rc = 0 ] || [ $rc = 10 ] || { echo "focal rule failed"; exit 1; }
  if [ $rc = 0 ]; then  # static camera: median(ViPE, GeoCalib, MoGe-2) focal, then ViPE with K fixed
    $PV -m bench.scene_recon.focal_compare geocalib --list $Lc --out $G/focal/geo_$c.json > $G/logs/focal_geo.log 2>&1
    python -m bench.scene_recon.focal_compare moge --list $Lc --out $G/focal/moge_$c.json > $G/logs/focal_moge.log 2>&1
    $FR focal
  fi
fi
if [ $ST != 0 ] && [ -f $G/focal/fixed_$c.json ]; then
  [ -f $G/vipe_fk/cams/$c.npz ] || $PV -m bench.world_vipe --list $Lc --out $G/vipe_fk --python $PY12 --vipe $V12 --depth \
    --fixed_focal_json $G/focal/fixed_$c.json --gpu $GPU --work $G/scratch/vipe_fk_$c ${VMPIX:+--chunk_mpix $VMPIX} ${VDEPTH:+--depth_align $VDEPTH} --stride $VSTRIDE --chunk_real $VCREAL --max_side $VMAXS --no_instance $VNOINST > $G/logs/vipe_fk.log 2>&1
  [ -f $G/vipe_fk/dense/$c.npz ] || $PY12 -m bench.videomimic.vipe_depth --vipe_out $G/vipe_fk --vipe $V12 --name $c > $G/logs/dense_fk.log 2>&1
  [ -f $G/geocalib_fk/$c.npz ] || $PV -m bench.world_geocalib --list $Lc --cams $G/vipe_fk/cams --out $G/geocalib_fk > $G/logs/geo_fk.log 2>&1
fi
$FR select $([ $ST = 0 ] && echo --src static)
log "cameras + depth"

# B: seg, FAB, contacts, the world chain
KW=$(python -c "import json; from fab.predictor import SIMPLE_PROFILE; k=dict(SIMPLE_PROFILE); k['smpl_fitter']='$FITTER'; k.update({'det_size': $DET_SIZE} if $DET_SIZE != 640 else {}); print(json.dumps(k))")
[ -n "$INIT_BOX" ] && TRACK_KW=$(python -c "import json, sys; k=json.loads(sys.argv[1] or '{}'); k.update(init_box=sys.argv[2], init_frame=int(sys.argv[3])); print(json.dumps(k))" "$TRACK_KW" "$INIT_BOX" "$INIT_FRAME")
[ -f $G/seg/${c}_seg.npz ] || python -m bench.scene_recon.clean_render seg --recon $G/vipe_sel/cams --dense $G/vipe_sel/dense --list $L --only $c --out $G/seg > $G/logs/seg.log 2>&1
[ -f $G/ours/fab/${c}_raw.npz ] || python -m bench.videomimic.recon fab --list $L --only $c --out $G/ours --vipe $G/vipe_sel --fab_ckpt $CK --detector deployed \
  --predictor_kw "$KW" ${SB:+--subject_boxes $G/subject_boxes} --load_workers ${FAB_LOAD_WORKERS:-4} --contact_feats $SPASS --fab_fast ${FAB_FAB_FAST:-0} --track $TRACK ${TRACK_KW:+--track_kw "$TRACK_KW"} > $G/logs/fab.log 2>&1
[ -f $G/ours/fab/${c}_raw.npz ] || { echo "FAB stage failed"; tail -n 5 $G/logs/fab.log; exit 1; }
log "seg + FAB"
W=$G/contact_work
if [ ! -f $G/contacts/${c}__0.contact.npz ]; then
  mkdir -p $W; export FAB_CONTACT_CLIPS=$L FAB_CONTACT_RAW="$G/ours/fab/{c}_raw.npz" FAB_CONTACT_CAMS="$G/vipe_sel/cams/{c}.npz" CLIPS_ONLY=$c
  python scripts/contact_clips.py prep --set clip --work $W > $W/prep.log 2>&1
  if [ $SPASS = 1 ] && [ -z "$CHEAD" ]; then  # the FAB stage's own contact features (the body model is the contact head)
    python scripts/contact_clips.py tracks --set clip --work $W > $W/extract.log 2>&1
  fi
  if ! ls $W/tracks/*.npz > /dev/null 2>&1; then  # the second FAB pass (default; or no features in the raw: e.g. a
    python -m train.extract_contact_tracks --source clips:$W/clips_spec.json --name tracks --fab_ckpt $HEAD --out $W \
      --workers ${FAB_LOAD_WORKERS:-4} >> $W/extract.log 2>&1  # TensorRT body engine exports no contact features)
  fi
  python -m train.train_contact_video --eval_ckpt $VID --test $W/tracks --pred_out $W/pred --test_cams $W/cams > $W/video.log 2>&1
  python scripts/contact_clips.py export --set clip --work $W --out $G/contacts > $W/export.log 2>&1
fi
[ -f $G/contacts/${c}__0.contact.npz ] || { echo "contacts failed"; tail -n 3 $W/*.log; exit 1; }
log contacts
[ $SREF = clip ] && SREF=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['fps'])" $L $c)  # the paper's smoothing: in frames
WA="--refine 0 --f_own 1.0 --vis_carve 1 --seg_dir $G/seg --geocalib $G/geo_sel --smooth_ref_fps $SREF"
[ -f $G/ours_v/recon/$c.npz ] || python -m bench.videomimic.recon world --list $L --only $c --out $G/ours_v --vipe $G/vipe_sel $WA > $G/logs/world_v.log 2>&1
MW=$G/scratch/mvs_$c; RU=""
if [ -n "$MVSR" ] && [ ! -f $G/vipe_mvs/dense/$c.npz ]; then  # link the existing workspace (symlinked files: mvs_depth never writes into it)
  src=$MVSR/$c; [ -d $src/$c/dense ] && src=$src/$c  # DIR/<name>/dense, or DIR/<name>/<name>/dense (a --work DIR/<name> run)
  rm -rf $MW; mkdir -p $MW; [ -d $src/dense ] && cp -rs $src $MW/$c; RU="--reuse 1"
fi
[ -f $G/vipe_mvs/dense/$c.npz ] || python -m bench.scene_recon.mvs_depth --vipe $G/vipe_sel --recon $G/ours_v/recon --list $L --only $c --seg_dir $G/seg \
  --out $G/vipe_mvs --work $MW $RU $([ $MVS = 0 ] && echo --min_parallax 1e9) > $G/logs/mvs.log 2>&1  # (--mvs 0: the no-parallax branch)
grep -h "\[mvs\]" $G/logs/mvs.log | cut -c1-160
[ -f $G/ours_m/recon/$c.npz ] || python -m bench.videomimic.recon world --list $L --only $c --out $G/ours_m --vipe $G/vipe_mvs $WA > $G/logs/world_m.log 2>&1
[ -f $G/joint/recon/$c.npz ] || python -m bench.scene_recon.joint --recon $G/ours_m/recon --vipe $G/vipe_mvs --list $L --only $c --out $G/joint \
  --contacts $G/contacts --snap 0 > $G/logs/joint.log 2>&1
[ -f $G/joint/recon/$c.npz ] || { echo "world chain failed"; tail -n 3 $G/logs/world_v.log $G/logs/mvs.log $G/logs/world_m.log $G/logs/joint.log; exit 1; }
grep -h "\[joint\]" $G/logs/joint.log | cut -c1-150
log "world chain"

# export: cameras + body in the joint-rescaled world, contacts, the scene
python - $G $c $OUT "$CK" "$FITTER" "$HEAD" "$VID" "$MVSR" $SREF <<'PY'
import hashlib, json, os, shutil, sys, numpy as np
G, c, X, ck, fitter, head, vid, mvsr = sys.argv[1:9]
sref = float(sys.argv[9])
md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
models = {k: {"path": os.path.realpath(f), "md5": md5(os.path.realpath(f))} for k, f in
          (("body", ck), ("smpl_fitter", fitter), ("contact_head", head), ("contact_video", vid))}
r = np.load(f"{G}/joint/recon/{c}.npz", allow_pickle=True)
fr = json.load(open(f"{G}/focal/{c}.json"))
fps_ = float(r["fps"])  # recon world: the smoothing sigmas are frames at sref fps, scaled by fps / sref
smooth = ("pelvis 3 / rotations 2 frames" if fps_ == sref else
          f"pelvis 3 / rotations 2 frames at {sref:g} fps = {3 * fps_ / sref:.3g} / {2 * fps_ / sref:.3g} frames at {fps_:g} fps")
np.savez(f"{X}/{c}.npz", frames=np.asarray(r["frames"]), T_world_cam=r["T_world_cam"].astype(np.float64), gravity_world=np.array([0.0, 0.0, -1.0]),
         K=r["K"], fps=r["fps"], pose=r["pose"], betas=r["betas"], trans=r["trans"], joints=r["joints"], valid=r["valid"],
         scale_fix=r["scale_fix"], joint=r["joint"], contacts=r["contacts"],
         notes=json.dumps({"pipeline": "FAB world pipeline (scripts/fab_world.sh): ViPE 1.2 + UniDepth, focal rule, GeoCalib z-up world, "
                           "FAB (the released profile + the SMPL fitter below), body smoothing (" + smooth + "), MVS scene, "
                           "joint human-scene step (the contact image head -> causal video model below, --snap 0)", "models": models,
                           "world": "z-up, gravity -z; T_world_cam camera -> world, metric after the joint step's scale", "camera": fr["vipe"],
                           "static_camera": fr["static"], "focal": fr.get("focal", fr.get("fx_vipe")),
                           "mvs_reuse": mvsr or None}))
for s in ("_scene.ply", "_hfield.npz"):
    if os.path.exists(f"{G}/joint/recon/{c}{s}"):
        shutil.copy(f"{G}/joint/recon/{c}{s}", f"{X}/{c}{s}")
print("[fab_world]", c, len(r["frames"]), "frames, scale_fix", float(r["scale_fix"]), "->", f"{X}/{c}.npz")
hf = f"{G}/joint/recon/{c}_hfield.npz"
hv = float(np.load(hf)["valid"].mean()) if os.path.exists(hf) else 0.0
if json.loads(str(r["joint"])).get("skipped") == "empty scene mesh" or hv == 0.0:  # docs/world.md, Outputs
    print(f"[fab_world] WARNING {c}: no scene was reconstructed (empty scene mesh, heightfield valid {hv:.2f}): no "
          f"{c}_scene.ply, the contact labels are measured against z = 0 and fab-retarget's reference can float. Usual "
          "causes: the background masked out (a plain backdrop taken for sky), beyond the 8 m fusion range (a distant "
          "subject, a long lens) or unusable depth (docs/troubleshooting.md)", flush=True)
lab = json.loads(str(r["contacts"]))
if not any(np.any(v) for v in lab.values()):
    print(f"[fab_world] WARNING {c}: no contact label on any frame ({', '.join(lab)})", flush=True)
PY
if [ $MOTION = learned ]; then  # FAB-M: the learned world-motion stage on the export (fab/world_motion.py)
  MMODEL=${MMODEL:-$FAB_WEIGHTS/shared/world_motion.pt}
  [ -f $MMODEL ] || { echo "$MMODEL: not found (FAB-M's model, shared/world_motion.pt in the FAB weights: README.md)"; exit 1; }
  python -m fab.world_motion $OUT/$c.npz --work $G --model $MMODEL --fps_mode $MFPS > $G/logs/world_motion.log 2>&1 || { echo "world motion failed"; tail -n 5 $G/logs/world_motion.log; exit 1; }
  log "world motion"
fi
if [ $REFINE = physics ]; then  # optional: the physical-plausibility post-process (docs/world.md, --refine physics)
  python -m bench.scene_recon.world_refine --world $OUT/$c.npz --hfield $OUT/${c}_hfield.npz --contacts $G/contacts/${c}__0.contact.npz \
    --scene $OUT/${c}_scene.ply --out $G/refine > $G/logs/refine.log 2>&1 || { echo "world refine failed"; tail -n 3 $G/logs/refine.log; exit 1; }
  grep -h "\[world_refine\]" $G/logs/refine.log | cut -c1-220
  cp $G/refine/$c.npz $OUT/$c.npz
  for s in _hfield.npz _scene.ply; do [ -f $G/refine/$c$s ] && cp $G/refine/$c$s $OUT/$c$s; done
  log "world refine"
fi
log "fab_world done"
