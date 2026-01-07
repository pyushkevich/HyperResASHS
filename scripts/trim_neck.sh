#!/bin/bash -e
# ------------------------------------
# Trim whitespace and lower resolution
# ------------------------------------

# Print usage by default
if [[ $# -lt 1 || $1 == "-h" || $1 == "-help" || $1 == "--help" ]]; then
  cat <<USAGETEXT
  trim_neck: Brain MRI neck removal tool

  Script by Paul Yushkevich and Sandhitsu Das. Modified by Philip Cook

  Usage:
    trim_neck [options] input_image output_image

  Options:
    -l <mm>        : Length (sup-inf) of the head that should be captured [180].
    -c <mm>        : Clearance above head that should be captured [10].
    -m <file>      : Location to save mask indicating trimmed region.
    -r             : Output replaces trimmed region with zeros, rather than cropping input.
    -w <dir>       : Location to store intermediate files. Default: [$TMPDIR]
    -d             : Verbose/debug mode
USAGETEXT
  exit 0
fi

# Default parameter values
HEADLEN=150
CLEARANCE=10
unset WORKINGDIR MASKOUT MASKTRIM DEBUG

# Read the options
while getopts "l:c:m:w:dr" opt; do
  case $opt in

    l) HEADLEN=$OPTARG;;
    c) CLEARANCE=$OPTARG;;
    m) MASKOUT=$OPTARG;;
    r) MASKTRIM=1;;
    w) WORKINGDIR=$OPTARG;;
    d) DEBUG=1;;
    \?) echo "Unknown option $OPTARG"; exit 2;;
    :) echo "Option $OPTARG requires an argument"; exit 2;;

  esac
done

shift $((OPTIND-1))

# Debugging
if [[ $DEBUG ]]; then
  set -x -e
  VERBOSE="-verbose"
fi

# Read script parameters
SOURCE=${1?}
TARGET=${2?}

# Create a temporary directory
if [[ ! $WORKINGDIR ]]; then
  if [[ ! $TMPDIR ]]; then
    TMPDIR=$(mktemp -d)
  fi
  WORKINGDIR=$TMPDIR
else
  mkdir -p $WORKINGDIR
fi

# Create a landmark file for background / foreground segmentation
cat > $WORKINGDIR/landmarks.txt << 'LANDMARKS'
40x40x40% 1
60x40x40% 1
40x60x40% 1
40x40x60% 1
60x60x40% 1
60x40x60% 1
40x60x60% 1
60x60x60% 1
3x3x3% 2
97x3x3% 2
3x97x3% 2
3x3x97% 2
97x97x3% 2
97x3x97% 2
3x97x97% 2
97x97x97% 2
LANDMARKS

# Intermediate images
RAS_IMAGE=$WORKINGDIR/source_ras.nii.gz
RAS_TRIMMED=$WORKINGDIR/trimmed_ras.nii.gz
RAS_RESULT=$WORKINGDIR/result_ras.nii.gz
RFMAP=$WORKINGDIR/rfmap.nii.gz
LEVELSET=$WORKINGDIR/levelset.nii.gz
MASK=$WORKINGDIR/mask.nii.gz
SLAB=$WORKINGDIR/slab.nii.gz

# Quick trim of the image
c3d $VERBOSE \
  $SOURCE -swapdim RAS -o $RAS_IMAGE \
  -smooth-fast 1vox -resample-mm 2x2x2mm -o $WORKINGDIR/dsample_ras.nii.gz -as T1 \
  -dup -steig 2.0 4x4x4 \
  -push T1 -dup -scale 0 -lts $WORKINGDIR/landmarks.txt 15 -o $WORKINGDIR/samples.nii.gz \
  -rf-param-patch 1x1x1 -rf-train $WORKINGDIR/myforest.rf -pop -rf-apply $WORKINGDIR/myforest.rf \
  -o $RFMAP

# Quick level set of the image
c3d $VERBOSE \
  $RFMAP -as R -smooth-fast 1vox -resample 50% -stretch 0 1 -1 1 -dup \
  $WORKINGDIR/samples.nii.gz -thresh 1 1 1 -1 -reslice-identity \
  -levelset-curvature 5.0 -levelset 300 -o $LEVELSET \
  -insert R 1 -reslice-identity -thresh 0 inf 1 0 -o $MASK

# Get the dimensions of the image and the trim amount
DIM=($(c3d $MASK -info-full | grep 'Dimen' | sed -e "s/.*\[//g" -e "s/,/ /g" -e "s/\].*//g"))
DIMX=${DIM[0]}
DIMY=${DIM[1]}
DIMZ=${DIM[2]}

# Amount to trim: 18cm = 90vox
REGZ=$((HEADLEN / 2 + CLEARANCE / 2))
TRIM=$((CLEARANCE / 2))

# Translate the RAI code into a region specification
REGCMD="0x0x0vox ${DIMX}x${DIMY}x${REGZ}vox"

# while true ; do
# :
# done
# Perform the trimming
c3d $VERBOSE $MASK \
  -dilate 1 ${DIMX}x${DIMY}x0vox -trim 0x0x${TRIM}vox \
  -region $REGCMD -thresh -inf inf 1 0 -o $SLAB -popas S \
  $SOURCE -as I -int 0 -push S -reslice-identity -trim 0vox -as SS -o $WORKINGDIR/slab_src.nii.gz \
  -push I -reslice-identity -o ${WORKINGDIR}/trimmed_input.nii.gz

TRIMREGIONMASK="${WORKINGDIR}/trim_mask.nii.gz"

# Make the mask
c3d $SOURCE -as I ${WORKINGDIR}/trimmed_input.nii.gz -thresh 0 0 1 1 -reslice-identity \
    -thresh 0.5 inf 1 0 -o ${TRIMREGIONMASK}

if [[ $MASKTRIM ]]; then
  c3d $SOURCE $TRIMREGIONMASK -multiply -o $TARGET
else
  c3d ${WORKINGDIR}/trimmed_input.nii.gz -o $TARGET
fi

# If mask requested, reslice mask to source space
if [[ -n $MASKOUT ]]; then
  c3d $TRIMREGIONMASK -o $MASKOUT
fi

