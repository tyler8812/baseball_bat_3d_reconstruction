VIDEONAME="4"
VIDEOFOLDER="input/baseball_swing_20220725/4"

python src/main.py \
--target_view1 "${VIDEOFOLDER}/${VIDEONAME}_view1_target.npy" \
--target_view2 "${VIDEOFOLDER}/${VIDEONAME}_view2_target.npy" \
--ref_points_view1 "${VIDEOFOLDER}/${VIDEONAME}_view1_coordinate.npy" \
--ref_points_view2 "${VIDEOFOLDER}/${VIDEONAME}_view2_coordinate.npy" \
--calib_file "${VIDEOFOLDER}/calib.npz" \
--start_frame 0 \
--end_frame 500 \
--frame_skip 10 \
--show \
--input_video "${VIDEOFOLDER}/${VIDEONAME}_view1.MP4" \
--output_fig "output/${VIDEONAME}/${VIDEONAME}.mp4"
# --ref_points_view1 "/Users/tylerchen/Desktop/碩士/baseball_bat_3d_construction/input/baseball_swing_20220628/2/1802_24_cam_0_coordinate.npy" \
# --ref_points_view2 "/Users/tylerchen/Desktop/碩士/baseball_bat_3d_construction/input/baseball_swing_20220628/2/1802_24_cam_1_coordinate.npy" \
# --calib_file "/Users/tylerchen/Desktop/碩士/baseball_bat_3d_construction/input/baseball_swing_20220628/2/calib.npz" \
# --calib_file "input/camera_calibration/C0180/camera_calibration.npz" \