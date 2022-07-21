
VIDEOFOLDER="../input/baseball_swing_20220628/1/1"
VIEW='1'
python get_img_coordinate.py \
--video "${VIDEOFOLDER}_view${VIEW}.MP4" \
--output "${VIDEOFOLDER}_view${VIEW}_coordinate.npy" \
--img_output "${VIDEOFOLDER}_view${VIEW}_img.png"
