def merge(
    start_frame,
    end_frame,
    frame_skip,
    target_view1,
    target_view2,
    ref_points_view1,
    ref_points_view2,
    calibration,
    input_video,
    output_fig=None,
    show=False,
):
    """
    output
        balls_frame_num: e.g., [[x, y ,z, frame number], [x, y ,z, frame number],...]
        court: e.g., [[x1, y1, z1], [x2, y2, z2],...]
    """
    court = None
    # if has start an end frame
    if start_frame is not None:
        print("Start from frame %d" % start_frame)
        target_view1 = target_view1[start_frame:]
        target_view2 = target_view2[start_frame:]

    if end_frame is not None:
        print("Run %d frames." % end_frame)
        target_view1 = target_view1[:end_frame]
        target_view2 = target_view2[:end_frame]
    frames = zip(target_view1, target_view2)

    target_frame_num = []

    """
    yolo[i]
        list([]) or list([[68, ['44', '973', '36', '30']]])
    detectron
        list([]) or list([(1330, 292), (1264, 430)])
    """
    proj_map_1, proj_map_2 = project_points(
        src_points_1=get_ref_points(ref_points_view1),
        src_points_2=get_ref_points(ref_points_view2),
        dst_points=get_ref_points(),
        dist=np.load(calibration)["dist_coefs"],
        mtx=np.load(calibration)["camera_matrix"],
    )

    for frame_num, frame in enumerate(frames):
        if frame_num % frame_skip != 0:
            continue
        view1, view2 = frame
        # print(view1, view2)
        # get targets position
        if len(view1) > 0 and len(view2) > 0:
            target_view1 = np.array(view1, dtype="int")
            target_view2 = np.array(view2, dtype="int")

            court, t1, t2 = draw2court(
                target_view1=target_view1,
                target_view2=target_view2,
                src_points_1=get_ref_points(ref_points_view1),
                src_points_2=get_ref_points(ref_points_view2),
                proj_map_1=proj_map_1,
                proj_map_2=proj_map_2,
            )
            t1.append(frame_num)
            t2.append(frame_num)
            target_frame_num.append(t1)
            target_frame_num.append(t2)

    target_frame_num = np.array(target_frame_num)
    target = target_frame_num.T

    # If there is no bat, just add court points.
    if target.shape[-1] == 0:
        court = np.array(get_ref_points()).T
        target = None

    print("total target points: %d" % target.shape[-1])
    if show:
        show_3D(
            input_video,
            court,
            target,
            frame_skip,
            end_frame,
            colorful=False,
            is_set_lim=True,
            add_court=True,
            court_category="baseball_bat",
            save_name=output_fig,
        )
    court = np.array(court).T
    return target_frame_num, court


def get_ref_points(ref_points="dst_points"):

    if ref_points == "dst_points":
        return [
            [0, 0, 0],
            [0, 0, 17],
            [0, 18, 17],
            [24, 0, 0],
            [24, 0, 17],
            [24, 18, 17],
        ]
    else:
        return np.load(ref_points)


if __name__ == "__main__":
    from pathlib import Path
    import argparse
    import numpy as np
    from draw2court import draw2court, show_3D, project_points

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--end_frame", default=None, help="End frame", type=int)
    parser.add_argument(
        "-s", "--start_frame", default=None, help="Start frame", type=int
    )
    parser.add_argument("--frame_skip", default=0, help="Start frame", type=int)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--calib_file", type=str)
    parser.add_argument("--ref_points_view1", type=str)
    parser.add_argument("--ref_points_view2", type=str)
    parser.add_argument("--target_view1", type=str)
    parser.add_argument("--target_view2", type=str)
    parser.add_argument("--output_fig", type=str)
    parser.add_argument("--input_video", type=str)

    args = parser.parse_args()

    end_frame = args.end_frame
    start_frame = args.start_frame
    frame_skip = args.frame_skip
    # is_re_run = args.re_run
    ref_points_view1 = args.ref_points_view1
    ref_points_view2 = args.ref_points_view2
    calibration = args.calib_file
    output_fig = args.output_fig
    input_video = args.input_video

    FPS = 120

    # load ball point in two views
    target_view1 = np.load(args.target_view1, allow_pickle=True)
    target_view2 = np.load(args.target_view2, allow_pickle=True)

    # compare the length of two view
    print(len(target_view1), len(target_view2))
    if len(target_view1) < len(target_view2):
        target_view2 = target_view2[: len(target_view1)]
    else:
        target_view1 = target_view1[: len(target_view2)]

    balls_frame_num, court = merge(
        start_frame,
        end_frame,
        frame_skip,
        target_view1,
        target_view2,
        ref_points_view1,
        ref_points_view2,
        calibration,
        input_video,
        output_fig=output_fig,
        show=args.show,
    )
