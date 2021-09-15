import numpy as np
from scipy.linalg import lstsq as lstsq
from scipy import interpolate as interpolate
from Kinetics_LBMC.utils.norm_vector import norm_vector as norm_vector
from matplotlib import pyplot as plt
from celluloid import Camera
from scipy.ndimage import gaussian_filter1d
import curvature
from math import pi


def normal_best_fitting_plan(points, points_ind, list_points_fitting, right_point, left_point):
    # points : coordinate of the points
    # points_ind list to know the position in points of the desired point using string
    # lists_points_fitting : list containing the point used for the approximation of the plane
    # right_point : String of one point of the segment situated on the right of the segment
    # left_point : String of one point of the segment situated on the left of the segment
    # These two last points are used to determine in which direction is the right to correct the
    # normal to the plane

    nb_frames = points.shape[2]
    Normal_plan = np.zeros((3, nb_frames))
    nb_point = len(list_points_fitting)

    Orientation = points[:, points_ind[right_point], :] - \
        points[:, points_ind[left_point], :]

    for frame in range(nb_frames):
        A = np.zeros((nb_point, 3))
        B = np.zeros((nb_point, 1))
        for ind_point, name_point in enumerate(list_points_fitting):
            A[ind_point, 0] = points[0, points_ind[name_point], frame]
            A[ind_point, 1] = points[1, points_ind[name_point], frame]
            A[ind_point, 2] = 1
            B[ind_point, 0] = points[2, points_ind[name_point], frame]
        value, res, rnk, toto = lstsq(A, B)
        # In order to have the 3 correct parameter we have to pass the z value from
        # right side of equation to the left side and so to remove 1
        value[2, 0] = -1

        # We project the norm to the plane on the global direction of the segement to
        # determine its correct orientation
        correction = np.sign(
            np.dot(value[:, 0], Orientation[:, frame]))
        Normal_plan[:, frame] = correction*value[:, 0]

    return Normal_plan


def extract_chord_parameters(points, points_ind, list_point_plane, list_points_spline, low_name, up_name, smoothness_spline=0.2, plot_chord=False, video=False, name_video='video_chord_demo.mp4'):
    # TODO Make the method for the 2 plane
    nb_frames = points.shape[2]

    nb_point = len(list_point_plane)

    normal_to_plane = np.zeros((3, 1, nb_frames))
    points_projected = np.zeros((3, len(list_points_spline), nb_frames))
    for frame in range(nb_frames):
        A = np.zeros((nb_point, 3))
        B = np.zeros((nb_point, 1))
        for ind_point, name_point in enumerate(list_point_plane):
            A[ind_point, 0] = points[0, points_ind[name_point], frame]
            A[ind_point, 1] = points[1, points_ind[name_point], frame]
            A[ind_point, 2] = 1
            B[ind_point, 0] = points[2, points_ind[name_point], frame]
        value, res, rnk, tot = lstsq(A, B)
        # In order to have the 3 correct parameter we have to pass the z value from
        # right side of equation to the left side and so to remove 1
        # normal_to_plane = np.zeros((1, 3))
        normal_to_plane[0, 0, frame] = value[0]
        normal_to_plane[1, 0, frame] = value[1]
        normal_to_plane[2, 0, frame] = -1

        # We suppose the equation a*x+b*y+d = z above
        # value[0] = a, value[1] = b et value[2]=d
        # Let's define a point of the plane : x=0 and y=0
        point_of_plane = np.zeros((3, 1))
        # then z = d
        point_of_plane[2, 0] = value[2]

        # to project the point on the plane we project point_to_project-point_of_plane on
        # the normal to the plane
        # Here it could be optimised doing it only on the first and last point
        ind_point_low_segment = list_points_spline.index(low_name)
        ind_point_up_segment = list_points_spline.index(up_name)

        for ind_point, name_point in enumerate(list_points_spline):
            point_to_project = points[:, points_ind[name_point], frame]
            vector_to_project = point_to_project-point_of_plane[:, 0]

            projection = np.dot(vector_to_project,
                                normal_to_plane[:, 0, frame])

            points_projected[:, ind_point, frame] = point_to_project - \
                projection*normal_to_plane[:, 0, frame]

        # in list_point_fitting the first and last point are supposed to be the extremities
        # of the segment as a results there are the extremity of the chord then points_projected[:,0,:] and
        # points_projected[:,-1,:] are the extremities. We also suppose that the first is the down part of the segment
        # and the last one the up part of the segment

    # We can now define a plane referential using the chord and the normal
    X_plane = norm_vector(
        points_projected[:, ind_point_up_segment, :]-points_projected[:, ind_point_low_segment, :])
    Y_plane = np.cross(normal_to_plane, X_plane, axisa=0, axisb=0, axisc=0)
    Y_plane = norm_vector(Y_plane)
    Z_plane = np.cross(X_plane, Y_plane, axisa=0, axisb=0, axisc=0)
    Z_plane = norm_vector(Z_plane)
    validation = np.zeros(nb_frames)
    coordinate_2D_point_XY = np.zeros((2, len(list_points_spline), nb_frames))
    coordinate_2D_point_XZ = np.zeros((2, len(list_points_spline), nb_frames))
    # fig =
    for ind_point, name_point in enumerate(list_points_spline):
        vector_to_project = points[:, points_ind[name_point],
                                   :]-points_projected[:, ind_point_low_segment, :]
        for ind_frame in range(nb_frames):
            # Defintion of the two plane
            coordinate_2D_point_XY[0, ind_point, ind_frame] = np.dot(
                X_plane[:, ind_frame], vector_to_project[:, ind_frame])
            coordinate_2D_point_XY[1, ind_point, ind_frame] = np.dot(
                Y_plane[:, 0, ind_frame], vector_to_project[:, ind_frame])

            coordinate_2D_point_XZ[0, ind_point, ind_frame] = np.dot(
                X_plane[:, ind_frame], vector_to_project[:, ind_frame])
            coordinate_2D_point_XZ[1, ind_point, ind_frame] = np.dot(
                Z_plane[:, 0, ind_frame], vector_to_project[:, ind_frame])

            validation[ind_frame] = np.dot(
                Y_plane[:, 0, ind_frame], X_plane[:, ind_frame])

    # Now it is possible to extract the position of the different point expressed in 2D (the plane)
    # We can imagine finding the lowest value between the point and use it as the position of the chord or

    # TODO How to know the direction ==> if it is a negative curve or a positive)
    # Possibility one add variable for the global orientation (like the anterior et posterior part)==>
    # We can suppose that the vertical is around Y in ISB

    # interpolation
    #x_new = np.arange(0,)
    if video:
        fig, axs = plt.subplots(1, 2)
        camera = Camera(fig)
    percentage_chord_XY = np.zeros(nb_frames)
    value_chord_XY = np.zeros(nb_frames)

    percentage_chord_XZ = np.zeros(nb_frames)
    value_chord_XZ = np.zeros(nb_frames)

    curvature_XY = np.zeros(nb_frames)
    cuvature_XZ = np.zeros(nb_frames)

    for frame in range(nb_frames):
        xc_XY, yc_XY, R_XY, residu_XY = curvature.least_squares_circle(
            coordinate_2D_point_XY[:, :, frame].T)
        xc_XZ, yc_XZ, R_XZ, residu_XZ = curvature.least_squares_circle(
            coordinate_2D_point_XZ[:, :, frame].T)
        # Calculation of the side of the curvature
        # As we have oriented our data previously, the sign of the curvature is directly given by the sign of y
        curvature_XY[frame] = np.sign(yc_XY)*1/R_XY
        cuvature_XZ[frame] = np.sign(yc_XZ)*1/R_XZ
        #frame_to_plot = 1000
        # We calculate on each axis the maximum distance between two point and we use this value
        # as the distance to put ref point from the mean of the point
        # ATTENTION le paramètre s à un impact très fort sur l'interpolation (smoothness)
        # Il faudrait voir s'il existe des recommandations à suivre
        tck_XY, u_XY = interpolate.splprep([coordinate_2D_point_XY[0, :, frame],
                                            coordinate_2D_point_XY[1, :, frame]], s=smoothness_spline)
        tck_XZ, u_XZ = interpolate.splprep([coordinate_2D_point_XZ[0, :, frame],
                                            coordinate_2D_point_XZ[1, :, frame]], s=smoothness_spline)

        unew_XY = np.arange(0, 1.01, 0.01)
        out_XY = interpolate.splev(unew_XY, tck_XY)
        unew_XZ = np.arange(0, 1.01, 0.01)
        out_XZ = interpolate.splev(unew_XZ, tck_XZ)

        # find which indice coorespond to the limit point
        temp_position_low_XY = np.amin(
            np.abs(out_XY[0]-coordinate_2D_point_XY[0, ind_point_low_segment, frame]))
        temp_position_up_XY = np.amin(
            np.abs(out_XY[0]-coordinate_2D_point_XY[0, ind_point_up_segment, frame]))
        temp_position_low_XZ = np.amin(
            np.abs(out_XZ[0]-coordinate_2D_point_XZ[0, ind_point_low_segment, frame]))
        temp_position_up_XZ = np.amin(
            np.abs(out_XZ[0]-coordinate_2D_point_XZ[0, ind_point_up_segment, frame]))

        position_low_XY = np.where(np.abs(
            out_XY[0]-coordinate_2D_point_XY[0, ind_point_low_segment, frame]) == temp_position_low_XY)[0]
        position_up_XY = np.where(np.abs(
            out_XY[0]-coordinate_2D_point_XY[0, ind_point_up_segment, frame]) == temp_position_up_XY)[0]

        position_low_XZ = np.where(np.abs(
            out_XZ[0]-coordinate_2D_point_XZ[0, ind_point_low_segment, frame]) == temp_position_low_XZ)[0]
        position_up_XZ = np.where(np.abs(
            out_XZ[0]-coordinate_2D_point_XZ[0, ind_point_up_segment, frame]) == temp_position_up_XZ)[0]

        # We look for the max only in the area of study
        max_XY = np.amax(
            np.abs(out_XY[1][position_low_XY[0]:position_up_XY[0]]))
        result_XY = np.where(np.abs(out_XY[1]) == max_XY)

        max_XZ = np.amax(
            np.abs(out_XZ[1][position_low_XZ[0]:position_up_XZ[0]]))
        result_XZ = np.where(np.abs(out_XZ[1]) == max_XZ)

        percentage_chord_XY[frame] = result_XY[0] / \
            (position_up_XY[0]-position_low_XY[0])
        value_chord_XY[frame] = out_XY[1][result_XY[0]]

        percentage_chord_XZ[frame] = result_XZ[0] / \
            (position_up_XZ[0]-position_low_XZ[0])
        value_chord_XZ[frame] = out_XZ[1][result_XZ[0]]
        # pdb.set_trace()
        dev_inflexion = False
        if dev_inflexion:
            infls = extract_inflexion_point(out_XY[0], out_XY[1])
            filtered_data = gaussian_filter1d(out_XY[1], 10)
            smooth_d1 = np.gradient(filtered_data)
            smooth_d2 = np.gradient(smooth_d1)
            fig_2, axs_2 = plt.subplots(1, 3)
            axs_2[0].plot(out_XY[0], out_XY[1], color='black')
            axs_2[0].plot(out_XY[0], filtered_data, color='red')
            for i, infl in enumerate(infls, 1):
                axs_2[0].axvline(x=infl, color='k',
                                 label=f'Inflection Point {i}')

            # Point utiliser pour le spline
            axs_2[0].plot(coordinate_2D_point_XY[0, :, frame],
                          coordinate_2D_point_XY[1, :, frame], 'bo')
            axs_2[1].plot(smooth_d1)
            axs_2[2].plot(smooth_d2)
            plt.show()
        # Interpolation
        if video:
            if frame % 10 == 0:
                infls = extract_inflexion_point(out_XY[0], out_XY[1])
                for i, infl in enumerate(infls, 1):
                    axs[0].axvline(x=infl, color='k',
                                   label=f'Inflection Point {i}')

                axs[0].plot(out_XY[0], out_XY[1], color='orange')
                # Position du ventre
                axs[0].plot([out_XY[0][result_XY[0]], out_XY[0][result_XY[0]]], [
                    coordinate_2D_point_XY[0, ind_point_low_segment, frame], out_XY[1][result_XY[0]]], 'r')
                # La corde
                axs[0].plot([coordinate_2D_point_XY[0, ind_point_low_segment, frame], coordinate_2D_point_XY[0, ind_point_up_segment, frame]],
                            [coordinate_2D_point_XY[1, ind_point_low_segment, frame], coordinate_2D_point_XY[1, ind_point_up_segment, frame]], 'r')
                # Point utiliser pour le spline
                axs[0].plot(coordinate_2D_point_XY[0, :, frame],
                            coordinate_2D_point_XY[1, :, frame], 'bo')
                axs[0].axis('equal')
                xlim_inf, xlim_sup = axs[0].get_xlim()
                ylim_inf, ylim_sup = axs[0].get_ylim()
                print(xlim_inf, xlim_sup, ylim_inf, ylim_sup)
                theta_fit = np.linspace(-pi, pi, 180)

                x_fit = xc_XY + R_XY*np.cos(theta_fit)
                y_fit = yc_XY + R_XY*np.sin(theta_fit)
                axs[0].plot(x_fit, y_fit, 'b-', label="fitted circle", lw=2)
                axs[0].set_xlim([xlim_inf, xlim_sup])
                axs[0].set_ylim([ylim_inf, ylim_sup])
                # XZ
                axs[1].plot(out_XZ[0], out_XZ[1], color='orange')
                # Position du ventre
                axs[1].plot([out_XZ[0][result_XZ[0]], out_XZ[0][result_XZ[0]]], [
                    coordinate_2D_point_XZ[0, ind_point_low_segment, frame], out_XZ[1][result_XZ[0]]], 'r')
                # La corde
                axs[1].plot([coordinate_2D_point_XZ[0, ind_point_low_segment, frame], coordinate_2D_point_XZ[0, ind_point_up_segment, frame]],
                            [coordinate_2D_point_XZ[1, ind_point_low_segment, frame], coordinate_2D_point_XZ[1, ind_point_up_segment, frame]], 'r')
                # Point utiliser pour le spline
                axs[1].plot(coordinate_2D_point_XZ[0, :, frame],
                            coordinate_2D_point_XZ[1, :, frame], 'bo')

                axs[1].axis('equal')
                camera.snap()
                plt.plot()
        # plt.show()
    if video:
        animation = camera.animate()
        animation.save(name_video)

    if plot_chord:
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(percentage_chord_XY)
        axs[1].plot(value_chord_XY)
        plt.show()
    return percentage_chord_XY, value_chord_XY, curvature_XY
# Chord
# - Calculate the best fitting plane OK see above
# - Project all point in this plane
# - Using the chord as X and its perpendicular as Y obtain the coordinate
# of all point in 2D
# - define a spline to approximate the curve of the different point
# - Find the position of the closest point percent by percent


def extract_inflexion_point(x, y):
    filtered_data = y
    #filtered_data = gaussian_filter1d(y, 10)
    smooth_d1 = np.gradient(filtered_data)
    smooth_d2 = np.gradient(smooth_d1)
    # find switching points
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    # The inflexion point are given at the indices we need to give them at their real position
    real_inflexion_position = x[infls]
    return real_inflexion_position
