
VIDEOFOLDER="../input/baseball_swing_20220628/4/4"
VIEW='2'
python get_img_coordinate.py \
--video "${VIDEOFOLDER}_view${VIEW}.MP4" \
--output "${VIDEOFOLDER}_view${VIEW}_coordinate.npy" \
--img_output "${VIDEOFOLDER}_view${VIEW}_img.png"
