set -eu
cd baseline

echo Baseline: creating symlinks.
rm -rf fmow_dataset
mkdir fmow_dataset
ln -s "`readlink -f "$1"`"/train fmow_dataset/train
ln -s "`readlink -f "$1"`"/val fmow_dataset/val
ln -s "`readlink -f "$2"`" fmow_dataset/"test"

echo Switching to Baseline virtualenv.
set +eu
source virtualenv/bin/activate
set -eu

cd code
echo Baseline: preparing dataset.
time python3 runBaseline.py -prepare
echo Baseline: testing CNN.
time python3 runBaseline.py -prebuilt -test_cnn
echo Baseline: preparing LSTM.
time python3 runBaseline.py -prebuilt -codes
echo Baseline: testing LSTM.
time python3 runBaseline.py -prebuilt -test_lstm
echo Baseline: testing CNN, no metadata.
time python3 runBaseline.py -prebuilt -test_cnn -nm

echo Leaving Baseline virtualenv.
set +eu
deactivate
set -eu
