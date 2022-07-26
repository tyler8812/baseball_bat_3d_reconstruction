import pdb
from turtle import shape
import numpy as np
import cv2
import matplotlib
import os

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation


def set_saved_video(output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def project_points(
    src_points_1,
    src_points_2,
    dst_points,
    dist,
    mtx,
):
    points_view1 = np.array([src_points_1]).astype("float32")
    points_view2 = np.array([src_points_2]).astype("float32")
    dst_points_pnp = np.array([dst_points]).astype("float32")
    # print(dst_points)
    # print(points_view1)
    # print(points_view2)
    retval1, rvec1, tvec1 = cv2.solvePnP(dst_points_pnp, points_view1, mtx, dist)
    r1, _ = cv2.Rodrigues(rvec1)
    retval2, rvec2, tvec2 = cv2.solvePnP(dst_points_pnp, points_view2, mtx, dist)
    r2, _ = cv2.Rodrigues(rvec2)

    # get view1 project to 3d point
    proj_map_1 = np.matmul(mtx, np.concatenate((r1, tvec1), axis=1))
    # get view2 project to 3d point
    proj_map_2 = np.matmul(mtx, np.concatenate((r2, tvec2), axis=1))

    return proj_map_1, proj_map_2


# target_view1: view 1 target position
# target_view2: view 2 target position
# src_points_1: view 1 target point
# src_points_2: view 2 target point
# dst_points: mapping 3d target point
# dist: distortion coefficients
# mtx: camera_matrix
def draw2court(
    target_view1, target_view2, src_points_1, src_points_2, proj_map_1, proj_map_2
):
    """
    detectron2
    target_view1 = [(x0, y0), (x1, y1)]
    target_view2 = [(x0, y0), (x1, y1)]

    yolo
    target_view1 = [468, 499, 20, 14]
    target_view2 = [1045, 444, 17, 15]

    view1ball = np.array([[468+20*0.5, 499+14*0.5]], dtype=np.float32)
    view2ball = np.array([[1045+17*0.5, 444+15*0.5]], dtype=np.float32) # read img
    """

    points_view1 = np.array(src_points_1).astype("float32")
    points_view2 = np.array(src_points_2).astype("float32")
    target_view1 = np.array(target_view1).astype("float32")
    target_view2 = np.array(target_view2).astype("float32")

    # read img
    points1 = np.concatenate((points_view1, target_view1), axis=0)
    points2 = np.concatenate((points_view2, target_view2), axis=0)
    # print(points1, points2)

    pts1 = np.transpose(points1)
    pts2 = np.transpose(points2)
    pts4D = cv2.triangulatePoints(proj_map_1, proj_map_2, pts1, pts2)

    pts4D = pts4D[:, :] / pts4D[-1, :]
    x, y, z, w = pts4D

    target1 = [x[-1], y[-1], z[-1]]
    target2 = [x[-2], y[-2], z[-2]]

    courtX = x[:-2]
    courtY = y[:-2]
    courtZ = z[:-2]
    court = [courtX, courtY, courtZ]
    return court, target1, target2


# draw the court of different balls
def drawCourt(court_category, ax):
    if court_category == "baseball_bat":
        # baseball home base
        points = np.array(
            [
                [0, 0, 0],
                [0, 0, 17],
                [0, 18, 17],
                [0, 18, 0],
                [24, 0, 0],
                [24, 0, 17],
                [24, 18, 17],
                [24, 18, 0],
            ]
        )
        court_edges = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]
        curves = points[court_edges]
        ax.plot(curves[:, 0], curves[:, 1], curves[:, 2], c="k", linewidth=1)
        court_edges = [1, 5]
        curves = points[court_edges]
        ax.plot(curves[:, 0], curves[:, 1], curves[:, 2], c="k", linewidth=1)
        court_edges = [2, 6]
        curves = points[court_edges]
        ax.plot(curves[:, 0], curves[:, 1], curves[:, 2], c="k", linewidth=1)
        court_edges = [3, 7]
        curves = points[court_edges]
        ax.plot(curves[:, 0], curves[:, 1], curves[:, 2], c="k", linewidth=1)

    return ax


def show_3D(
    input_video,
    court,
    target,
    frame_skip,
    end_frame,
    alpha=0.2,
    colorful=False,
    add_court=False,
    isShow=True,
    save_name=None,
    is_set_lim=True,
    court_category="baseball_bat",
):
    if save_name is not None:
        video = set_saved_video(save_name, (2688, 756))

    court_x, court_y, court_z = court[0], court[1], court[2]
    target_x, target_y, target_z = target[0], target[1], target[2]

    print(input_video)
    frame_count = -1
    count = -1
    cap = cv2.VideoCapture(input_video)
    history = 3
    while cap.isOpened():
        fig = plt.figure(figsize=(19.2, 10.8))
        gs = gridspec.GridSpec(6, 6)
        ax = plt.subplot(gs[:, :], projection="3d")
        if is_set_lim:
            ax.set_xlim(-50, 350)
            ax.set_ylim(-200, 100)
            ax.set_zlim(0, 300)

        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        frame_count += 1
        if frame_count == end_frame:
            break
        ret, frame = cap.read()
        if frame_count % frame_skip != 0:
            continue
        else:
            count += 1
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        ax.scatter(court_x, court_y, court_z, marker="o", alpha=alpha)
        if count >= history - 1:
            ax.scatter(
                target_x[count * 2 : (count + 1) * 2],
                target_y[count * 2 : (count + 1) * 2],
                target_z[count * 2 : (count + 1) * 2],
                color="r",
                marker="o",
                alpha=alpha,
            )
            ax.scatter(
                target_x[(count - history + 1) * 2 : count * 2],
                target_y[(count - history + 1) * 2 : count * 2],
                target_z[(count - history + 1) * 2 : count * 2],
                color="b",
                marker="o",
                alpha=alpha,
            )
            ax.plot(
                [target_x[count * 2], target_x[count * 2 + 1]],
                [target_y[count * 2], target_y[count * 2 + 1]],
                [target_z[count * 2], target_z[count * 2 + 1]],
                color="r",
            )
            for i in range(1, history):
                ax.plot(
                    [target_x[(count - i) * 2], target_x[(count - 1) * 2 + 1]],
                    [target_y[(count - i) * 2], target_y[(count - 1) * 2 + 1]],
                    [target_z[(count - i) * 2], target_z[(count - 1) * 2 + 1]],
                    color="b",
                )
        else:

            ax.scatter(
                target_x[count * 2 : (count + 1) * 2],
                target_y[count * 2 : (count + 1) * 2],
                target_z[count * 2 : (count + 1) * 2],
                color="b",
                marker="o",
                alpha=alpha,
            )
            ax.plot(
                [target_x[count * 2], target_x[count * 2 + 1]],
                [target_y[count * 2], target_y[count * 2 + 1]],
                [target_z[count * 2], target_z[count * 2 + 1]],
                color="b",
            )

        if add_court:
            drawCourt(court_category, ax)

        fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        merge_image = cv2.hconcat([frame, img])
        resize_size = (int(merge_image.shape[1] * 0.7), int(merge_image.shape[0] * 0.7))
        merge_image = cv2.resize(merge_image, resize_size)
        print(resize_size)
        cv2.imshow("frame", merge_image)
        if save_name is not None:
            video.write(merge_image)
        if cv2.waitKey(1) == ord("q"):
            break
    plt.close(fig)
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    return

    def split_targets(i):
        if colorful:
            color = list(range(1, len(x[:i]) + 1))

        ax.scatter(court_x, court_y, court_z, marker="o", alpha=alpha)
        ax.scatter(
            target_x[i * 2 : (i + 1) * 2],
            target_y[i * 2 : (i + 1) * 2],
            target_z[i * 2 : (i + 1) * 2],
            color="b",
            marker="o",
            alpha=alpha,
        )
        ax.plot(
            [target_x[i * 2], target_x[i * 2 + 1]],
            [target_y[i * 2], target_y[i * 2 + 1]],
            [target_z[i * 2], target_z[i * 2 + 1]],
            color="b",
        )
        print("output/video2frame/4_view1/frame{}.png".format(i))
        image = plt.imread("output/video2frame/4_view1/frame{}.png".format(i))
        ax_video.imshow(image)

        if add_court:
            drawCourt(court_category, ax)

        if is_set_lim:
            ax.set_xlim(-50, 350)
            ax.set_ylim(-200, 100)
            ax.set_zlim(0, 300)

        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")

    def rotate(angle):
        ax.view_init(azim=angle, elev=angle / 4)

    if save_name is not None:
        if not os.path.isdir(save_name):
            os.mkdir(save_name)
        print("saving......")
        split_target_and_save(save_name, split_targets, fig, len(target_x))
        # plt.savefig(save_name + "/figure", transparent=True)
        # rotate_3d_plot_and_save(save_name, rotate, fig)
        print("finsih saving......")

    else:
        color = ["b"]
        if colorful:
            color_list = ["b", "g", "r", "c", "m", "y", "k", "w"]
            color = []
            for i in range(len(target_x) // 2):
                color.append(color_list[i % len(color_list)])
                color.append(color_list[i % len(color_list)])

        ax.scatter(court_x, court_y, court_z, marker="o", alpha=alpha)
        ax.scatter(target_x, target_y, target_z, color="b", marker="o", alpha=alpha)
        for i in range(0, len(target_x), 2):
            ax.plot(
                [target_x[i], target_x[i + 1]],
                [target_y[i], target_y[i + 1]],
                [target_z[i], target_z[i + 1]],
                color="b",
            )
        image = plt.imread(
            "/Users/tylerchen/Desktop/碩士/baseball_bat_3d_construction/output/video2frame/4_view1/frame2.png"
        )
        ax_video.imshow(image)
        if add_court:
            drawCourt(court_category, ax)
        if is_set_lim:
            ax.set_xlim(-50, 350)
            ax.set_ylim(-200, 100)
            ax.set_zlim(0, 300)
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")

    if isShow:
        plt.show()

    plt.close(fig)


# rotate the plot and save as gif
def rotate_3d_plot_and_save(file, rotate, fig):

    rot_animation = animation.FuncAnimation(
        fig, rotate, frames=np.arange(0, 362, 2), interval=50
    )
    rot_animation.save(file + "/rotate.gif", dpi=80, writer="imagemagick")


# split the ball in the plot and save as gif
def split_target_and_save(file, split_targets, fig, target_count):
    ani = animation.FuncAnimation(
        fig, split_targets, frames=range(0, target_count, 1), interval=50
    )
    ani.save(file + "/ball_split.gif", dpi=80, writer="imagemagick")
