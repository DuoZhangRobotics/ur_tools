import socket
import time

import numpy as np
import cv2

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO
from ur5 import UR5


class UR5_VGA10(UR5):
    def __init__(self):
        self.rtde_frequency = 100.0
        self.joint_acc = 0.4
        self.joint_vel = 0.2
        self.joint_acc_slow = 0.2
        self.joint_vel_slow = 0.1
        self.joint_acc_fast = 1.3
        self.joint_vel_fast = 0.9
        self.tool_vel_fast = 0.6
        self.tool_acc_fast = 0.8
        self.tool_vel = 0.3
        self.tool_acc = 0.4
        self.tool_vel_slow = 0.02
        self.tool_acc_slow = 0.2
        self.tool_acc_stop = 1.0
        self.pre_grasp_config = [74.44, -91.23, 87.61, -86.29, -89.51, -105.44]
        self.pre_grasp_config = [np.deg2rad(j) for j in self.pre_grasp_config]
        self.take_photo_config = [75.25, -100.69, 68.68, -57.90, -89.57, -101.69]
        self.take_photo_config = [np.deg2rad(j) for j in self.take_photo_config]
        self.drop_joint_config = [-45.53, -120.04, 106.72, -76.62, -89.46, -135.45]
        self.drop_joint_config = [np.deg2rad(j) for j in self.drop_joint_config]
        self.take_photo_yaw = np.deg2rad(0)
        self.force_threshold = 15

        self.pixel_size = 0.001

        # check the control script for parameters
        self.vg_cmd_in_reg = 19
        self.vg_cmd_out_reg = 19
        self.vg_threshold = 30

        # Use external UR cap, on the panel -> program, need to have
        # BeforeStart -> script: rtde_initialize_vg.script
        # Robot Program -> script: rtde_control_vg.script
        _ip = "172.17.139.100"
        self.rtde_c = RTDEControl(_ip, self.rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
        self.rtde_r = RTDEReceive(_ip, self.rtde_frequency, use_upper_range_registers=False)
        self.rtde_i = RTDEIO(_ip, self.rtde_frequency, use_upper_range_registers=False)

        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
        # self.go_take_photo()
        self.vg_grip(0.5)
        self.vg_release(0.5)

    def calibrate_marker(self, camera, debug=False):
        from scipy.optimize import least_squares
        from real_camera.camera import reprojection_error, reprojection_error2d

        # calibration_joint_configs = [
        #     [72.44, -35.76, 77.23, -159.99, -144.74, -29.57],
        #     [68.16, -47.30, 104.98, -181.18, -142.97, -35.73],
        #     [62.10, -53.52, 125.33, -204.08, -144.98, -50.05],
        #     [59.06, -63.32, 148.2, -217.08, -137.29, -47.02],
        #     [80.52, -70.59, 150.27, -186.79, -148.27, -15.95],
        #     [74.59, -61.02, 131.87, -186.53, -146.00, -26.14],
        #     [75.54, -56.53, 113.26, -174.25, -137.85, -18.92],
        #     [80.31, -38.21, 83.38, -177.51, -144.01, -14.17],
        #     [80.16, -31.53, 64.43, -193.47, -131.98, -30.17],
        # ]
        calibration_joint_configs = [
            [
                0.7645026445388794,
                -0.9935296338847657,
                2.4979055563556116,
                -3.770919462243551,
                -2.469565455113546,
                0.7427856922149658,
            ],
            [
                0.9382138252258301,
                -0.9956174057773133,
                2.268116299306051,
                -3.336738248864645,
                -2.5661326090442103,
                0.9910783171653748,
            ],
            [
                1.0641804933547974,
                -0.9234558504870911,
                1.9706767241107386,
                -2.9219685993590296,
                -2.6136792341815394,
                1.2129168510437012,
            ],
            [
                1.1368317604064941,
                -0.8196975153735657,
                1.6874473730670374,
                -2.6200596294798792,
                -2.629838291798727,
                1.3537166118621826,
            ],
            [
                1.1837865114212036,
                -0.6223752659610291,
                1.3937414328204554,
                -2.962942739526266,
                -2.4870105425464075,
                0.9429416060447693,
            ],
            [
                1.2870022058486938,
                -0.6908366841128846,
                1.3310940901385706,
                -2.710564275781149,
                -2.54116660753359,
                1.0916932821273804,
            ],
            [
                1.2210423946380615,
                -0.9103548687747498,
                1.8060615698443812,
                -3.0454260311522425,
                -2.509538237248556,
                0.9948655366897583,
            ],
            [
                1.1406010389328003,
                -1.0488423866084595,
                2.129860226308004,
                -3.316247125665182,
                -2.4642794767962855,
                0.8874495029449463,
            ],
            [
                1.106325626373291,
                -1.4063248199275513,
                2.5118542353259485,
                -3.1368500194945277,
                -2.2507551352130335,
                1.0212348699569702,
            ],
            [
                1.2530286312103271,
                -1.160969303255417,
                2.073494259511129,
                -2.8278142414488734,
                -2.306936566029684,
                1.1982682943344116,
            ],
            [
                1.322914719581604,
                -0.9261545699885865,
                1.6310508886920374,
                -2.559408803979391,
                -2.3269031683551233,
                1.287400722503662,
            ],
            [
                1.4421395063400269,
                -0.9308705490878602,
                1.6035502592669886,
                -2.4139167271056117,
                -2.3538203875171106,
                1.4489045143127441,
            ],
            [
                1.4118423461914062,
                -1.146785483961441,
                1.9934352079974573,
                -2.6164094410338343,
                -2.3495777289019983,
                1.4085679054260254,
            ],
            [
                1.356427550315857,
                -1.3796207022717972,
                2.374202553425924,
                -2.81656612972402,
                -2.3383124510394495,
                1.3343138694763184,
            ],
        ]

        rvec = cv2.Rodrigues(camera.camera_pose[:3, :3])[0].flatten().tolist()
        tvec = camera.camera_pose[:3, 3]
        camera_pose_esitimate = [*rvec, *tvec]
        marker_offset_esitimate = np.loadtxt(
            "real_camera/marker_offset.txt", delimiter=" "
        )  # [0.00545842, -0.00811971, -0.36624882]
        marker_points = []
        marker_pixels = []
        marker_depths = []
        ee_poses = np.zeros((len(calibration_joint_configs), 4, 4))

        # Collect marker pos
        print("Collecting marker pos...")
        for i, joint_config in enumerate(calibration_joint_configs):
            # joint_config = [np.deg2rad(j) for j in joint_config]
            self.rtde_c.moveJ(joint_config, self.joint_acc, self.joint_vel)
            time.sleep(1)

            color_img, depth_img = camera.get_data()
            marker_pos = camera.read_tag_arm(color_img)
            depth = depth_img[int(marker_pos[1]), int(marker_pos[0])]
            marker_points.append(camera.pos_in_image_to_camera(marker_pos, depth))
            marker_pixels.append(marker_pos)
            marker_depths.append(depth)

            ee_pose = self.rtde_r.getActualTCPPose()
            ee_poses[i, :3, :3] = cv2.Rodrigues(ee_pose[3:])[0]
            ee_poses[i, :3, 3] = ee_pose[:3]
            ee_poses[i, 3, 3] = 1

            print(f"marker_pos: {marker_pos}, depth: {depth} ee_pose: {ee_pose}")

            if debug:
                gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected_img_points = camera.aruco_detector_arm.detectMarkers(gray)
                if len(corners) > 0:
                    cv2.aruco.drawDetectedMarkers(color_img, corners, ids)
                    for i in range(0, len(ids)):
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners[i], camera.tag_length_ee, camera.intrinsics, camera.distortion_coeffs
                        )
                        cv2.drawFrameAxes(color_img, camera.intrinsics, camera.distortion_coeffs, rvec, tvec, 0.1)
                cv2.imshow("frame color", color_img)
                cv2.imshow("frame depth", depth_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
                input("waiting for next marker...")
        if debug:
            cv2.destroyAllWindows()

        print(np.array(ee_poses).tolist())
        print(np.array(marker_points).tolist())
        print(np.array(marker_pixels).tolist())
        print(np.array(marker_depths).tolist())

        # ee_poses = [[[-0.5021636477407653, 0.00935936135382873, -0.8647219629700271, -0.35938347291310657], [0.05643837640903733, 0.9981642995283727, -0.021971363529492898, -0.7391539316240249], [0.8629289545240677, -0.059836723691072596, -0.5017700528548091, 0.042316337634871484], [0.0, 0.0, 0.0, 1.0]], [[-0.49859967840549924, 0.011817681735491657, -0.8667518117040954, -0.35995799332211975], [0.056361498358396885, 0.998233191050034, -0.018811639717554653, -0.5943312875620637], [0.8649981168746724, -0.05823090832595076, -0.49838481028097703, 0.043361607194438836], [0.0, 0.0, 0.0, 1.0]], [[-0.42296468695115524, 0.0213130782808098, -0.9058954830920106, -0.371768713417673], [-0.021116519787240124, 0.9992200275925474, 0.033368084302580155, -0.44832144114879285], [0.9059000862039844, 0.03324288122508446, -0.4221847281267626, 0.06761346531458287], [0.0, 0.0, 0.0, 1.0]], [[-0.5021298724235529, 0.00927780687213366, -0.8647424550231991, -0.3064682822441284], [0.05654553217079726, 0.9981548390680252, -0.02212509969707947, -0.33203274083350803], [0.8629415936269543, -0.06000699579822141, -0.5017279805271423, 0.06305734229322847], [0.0, 0.0, 0.0, 1.0]], [[-0.5022496342741677, 0.009155038557294482, -0.8646742104055603, -0.1857728041572238], [0.05641815545201151, 0.9981603314491293, -0.02220235250501216, -0.33041388704432806], [0.8628802330606822, -0.05993444744367472, -0.5018421718058018, 0.051173523810470384], [0.0, 0.0, 0.0, 1.0]], [[-0.5022481316444927, 0.009083877814665, -0.8646758337223637, -0.25099389041585163], [0.05637870994028727, 0.9981612442236083, -0.02226143740748445, -0.45073753372393954], [0.8628836858610118, -0.059930073367439086, -0.5018367573036955, 0.051188289587086044], [0.0, 0.0, 0.0, 1.0]], [[-0.6039020968113081, 0.016481847168876004, -0.7968880763198848, -0.2510322875502259], [0.1879430860415343, 0.9745393682242406, -0.12227189452315615, -0.5915177274856822], [0.7745835357636316, -0.22360985777688605, -0.5916240171163701, 0.051180986123547836], [0.0, 0.0, 0.0, 1.0]], [[-0.5016465072955651, -0.00849695855711527, -0.8650309725168452, -0.2510758450471293], [0.4342443792893831, 0.8623673614301661, -0.26029666343530367, -0.7770337105500178], [0.7481862072864875, -0.5062117497996782, -0.4289138183726529, 0.08068659208233886], [0.0, 0.0, 0.0, 1.0]]]
        # marker_points = [[-0.24728347532145192, -0.2599256868390133, 0.6310000419616699], [-0.10629443981410375, -0.25860679950852444, 0.6290000081062317], [0.019816093056654605, -0.2534090870701739, 0.6290000081062317], [0.15371178691295007, -0.2204309040809138, 0.5800000429153442], [0.15192035963309133, -0.10966454570037608, 0.5300000309944153], [0.03445849576224462, -0.16804157488834798, 0.5649999976158142], [-0.0680791151709192, -0.2075143281152954, 0.550000011920929], [-0.2001848315407918, -0.1719608824470122, 0.5640000104904175]]
        # ee_poses = np.array(ee_poses)
        # marker_points = np.array(marker_points)
        # offset_homogeneous = np.append(marker_offset_esitimate, 1)
        # points = camera.pos_in_camera_to_robot(marker_points)
        # for point, ee_pose in zip(points, ee_poses):
        #     print(point)
        #     print(np.dot(ee_pose, offset_homogeneous))
        # ee_poses = [[[0.3141149920443394, 0.4555215738088817, -0.832965706117403, -0.4061612865834665], [0.9493474355867503, -0.14290963933136921, 0.2798504628029738, -0.24086615806262965], [0.008439094610517228, -0.8786790829312241, -0.47733829817153495, 0.043192082702131096], [0.0, 0.0, 0.0, 1.0]], [[0.31412573350026307, 0.45558482219641605, -0.8329270636359355, -0.4061435559606071], [0.9493427300316399, -0.14281761171481322, 0.2799133986077654, -0.3419694374434223], [0.008567641973996731, -0.8786612541635923, -0.4773688259015963, 0.043157406389307], [0.0, 0.0, 0.0, 1.0]], [[0.31425381954232967, 0.45551186588777626, -0.8329186496522293, -0.4061773193923118], [0.9493003745218068, -0.14286474040458394, 0.2800329710621372, -0.4561522520653294], [0.008563634497909423, -0.8786914168151223, -0.477313375236457, 0.04316373626994374], [0.0, 0.0, 0.0, 1.0]], [[0.31422696905926306, 0.45548172500705586, -0.8329452623674773, -0.4061853080922388], [0.9493108922126802, -0.143009377425788, 0.2799234679240984, -0.5509591278212325], [0.008381040639092951, -0.8786835130767422, -0.47733116596857633, 0.0432150235827832], [0.0, 0.0, 0.0, 1.0]], [[0.011631429835668505, 0.49104042492364186, -0.8710591317073642, -0.41298914907563605], [0.9958899094093586, -0.08393968848077688, -0.03402083235856518, -0.7479419910154441], [-0.089822036141458, -0.8670832888417117, -0.4899985428905307, 0.09384770020312351], [0.0, 0.0, 0.0, 1.0]], [[0.01159075031111724, 0.491048714386731, -0.8710550009077291, -0.3457419121115261], [0.9958735718835889, -0.08408156082490768, -0.03414849854780688, -0.7479881744477632], [-0.09000824035080096, -0.8670648483408759, -0.4899970055424485, 0.09392403299867837], [0.0, 0.0, 0.0, 1.0]], [[0.011555552236847719, 0.4910860473078881, -0.8710344214507348, -0.3457433137351501], [0.9958850093008339, -0.0839565481800922, -0.03412251848210962, -0.609430921883464], [-0.08988613609661694, -0.8670558183630386, -0.49003539706884625, 0.09386990866518019], [0.0, 0.0, 0.0, 1.0]], [[0.011661485398696644, 0.4910957111917942, -0.8710275611066056, -0.3457672788222579], [0.9958964375394623, -0.08388581381938716, -0.03396256668844183, -0.4973323265945294], [-0.08974572666430264, -0.8670571911292155, -0.4900587024598289, 0.0938142194129321], [0.0, 0.0, 0.0, 1.0]], [[0.03406945241511089, 0.6940972437246729, -0.7190746057718538, -0.2522378150486881], [0.9958741712786956, -0.08412646097488535, -0.03402019317717814, -0.36928139412860483], [-0.0841065240757572, -0.7149487777579968, -0.694099659840171, 0.059716118160299675], [0.0, 0.0, 0.0, 1.0]], [[0.0339852036850635, 0.6940834161256388, -0.7190919394554829, -0.2522538988656201], [0.9958856300885728, -0.08399575165891729, -0.03400772686228681, -0.5296079086627044], [-0.08400486720167566, -0.7149775696919566, -0.6940823129311193, 0.0597168963191487], [0.0, 0.0, 0.0, 1.0]], [[0.03397078677699582, 0.6940644692128899, -0.7191109081511566, -0.25227219089597613], [0.9958856315394113, -0.08399083166527911, -0.03401983376682366, -0.6727530957563583], [-0.08401068110127102, -0.7149965403919127, -0.6940620668848689, 0.05975240389184555], [0.0, 0.0, 0.0, 1.0]], [[0.033961640210920505, 0.6941039053470468, -0.7190732755263277, -0.17452627988471808], [0.9958847325193669, -0.08399293238403939, -0.03404095833491233, -0.6804075877302836], [-0.08402503513247128, -0.7149580098799566, -0.6941000198670787, 0.05966764183307041], [0.0, 0.0, 0.0, 1.0]], [[0.03383410434725642, 0.6940913094008936, -0.7190914459212904, -0.17448432123359997], [0.9959002953155796, -0.08380958728717136, -0.03403755086805682, -0.5566174294259468], [-0.08389192555520775, -0.7149917533041252, -0.6940813623299018, 0.059660790108703426], [0.0, 0.0, 0.0, 1.0]], [[0.03392810302100735, 0.6940906101906625, -0.7190876919198032, -0.174534245314742], [0.9958803111510772, -0.08401765142878435, -0.034109237898623634, -0.42020143023819895], [-0.08409096079264122, -0.7149680126366109, -0.6940817323769769, 0.059721630979193], [0.0, 0.0, 0.0, 1.0]]]
        # marker_points = [[0.14629578040385446, -0.0936549994337702, 0.5222499966621399], [0.042551349681353545, -0.09629896061052445, 0.5247500538825989], [-0.07387007593120988, -0.10017192557136897, 0.5297500491142273], [-0.17149464782703597, -0.10385733383351181, 0.5335000157356262], [-0.25505650539789204, -0.10994880213700886, 0.48075002431869507], [-0.25809762158404204, -0.0430887528895884, 0.468250036239624], [-0.11497412133102099, -0.03907633841792102, 0.46250003576278687], [-0.0007080611816045858, -0.03574669056355934, 0.4595000147819519], [0.1290080223458994, -0.00502280038806393, 0.4102500081062317], [-0.036233500754845285, -0.009688001100329652, 0.4140000343322754], [-0.18424506732841395, -0.013585475962176994, 0.4202500283718109], [-0.196621654348785, 0.06465032275435667, 0.4072500169277191], [-0.06746066891688075, 0.06736329286941913, 0.40175002813339233], [0.0735941873195606, 0.07128374958568284, 0.3967500329017639]]

        print("Start optimization...")
        for _ in range(2):
            # Run the optimization
            x0 = np.hstack([camera_pose_esitimate, marker_offset_esitimate])
            # result = least_squares(
            #     reprojection_error2d,
            #     x0,
            #     args=(ee_poses, marker_pixels, camera.intrinsics),
            #     method="lm",
            # )
            result = least_squares(
                reprojection_error,
                x0,
                args=(ee_poses, marker_points),
                method="lm",
            )

            # Extract optimized values
            optimized_params = result.x
            optimized_camera_pose_params = optimized_params[:6]
            optimized_camera_pose = np.eye(4)
            optimized_camera_pose[:3, 3] = optimized_camera_pose_params[3:]
            optimized_camera_pose[:3, :3] = cv2.Rodrigues(optimized_camera_pose_params[:3])[0]
            optimized_marker_offset = optimized_params[6:]

            print(optimized_camera_pose)
            print(optimized_marker_offset)
            camera_pose_esitimate = optimized_camera_pose_params
            marker_offset_esitimate = optimized_marker_offset

        # Update camera pose
        camera.camera_pose = optimized_camera_pose
        camera.camera_pose_inv = np.linalg.inv(camera.camera_pose)
        camera.configs["pose"] = camera.camera_pose

        # Compute calibration error
        image_points = np.array(marker_pixels)
        image_points_depth = np.array(marker_depths)
        offset_homogeneous = np.append(marker_offset_esitimate, 1)
        measured_points = np.dot(ee_poses, offset_homogeneous)[:, :3]
        observed_points = camera.pos_in_image_to_robot(image_points, image_points_depth)
        error = camera.compute_calibration_error(measured_points, observed_points)
        print("Mean error: ", np.mean(error))
        print("Std error: ", np.std(error))
        print("Per marker error: ", error)

        # Save camera optimized offset and camera pose
        print("Saving...")
        np.savetxt("real_camera/camera_pose.txt", camera.camera_pose, delimiter=" ")
        np.savetxt("real_camera/marker_offset.txt", marker_offset_esitimate, delimiter=" ")
        print(camera.camera_pose)
        print(marker_offset_esitimate)
        print("Done.")

        self.go_home()

        return camera.camera_pose, marker_offset_esitimate

    def handeye_calibration(self, camera):

        def detect_corners(img, checkerboard_size, criteria):
            ret, corners = cv2.findChessboardCorners(img, checkerboard_size)
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners if ret else None

        # Get the pose of the checkerboard w.r.t the camera
        def get_camera_checkerboard_pose(img, checkerboard_size, camera_matrix, dist_coeffs, obj_points, criteria):
            corners = detect_corners(img, checkerboard_size, criteria)
            if corners is not None:
                ret, rvecs, tvecs = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)
                if ret:
                    pose = np.hstack((rvecs.flatten(), tvecs.flatten()))
                    return pose
            return None

        # constants
        checkboard_size = (9, 6)
        square_size = 0.025  # Size of a square in the checkerboard (meters)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        obj_points = np.zeros((checkboard_size[0] * checkboard_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0 : checkboard_size[0], 0 : checkboard_size[1]].T.reshape(-1, 2) * square_size
        calibration_ee_configs = [[-0.008838529275751498, -0.3543261178563034, 0.48518985242603885, 4.471257551501001e-05, -0.00041900060665787535, 1.570308860128313], [0.22845840093759467, -0.2483178163944671, 0.48516511994302863, -0.003995623139914869, 0.34739246140658786, 1.5809826460093501], [0.22844628750190787, -0.31279487906888814, 0.4851864830608382, 0.24233273040725214, 0.18323447627520648, 1.6123096722570995], [0.24088115401806443, -0.3890905751730622, 0.4851651459149484, 0.21899703735198903, 0.09513913638895942, 1.6217893857653576], [0.24090895262306766, -0.46740912838351006, 0.48518389538260376, 0.4937246612155922, 0.22563355732426488, 1.5760659254943064], [0.10164896758548822, -0.46742338304705167, 0.4851636409258532, 0.2092707249740482, -0.08628377129429869, 1.608448864313463], [0.1016467150687567, -0.36316934354857305, 0.4851670639307584, 0.18578142793365812, 0.060116025827251784, 1.5941967086107787], [0.10164254102420471, -0.27185100356361425, 0.4851641569640334, 0.12778499603351956, 0.1969746909958915, 1.5774156898072813], [-0.007082292827432104, -0.2718356785157909, 0.48519926944543107, -0.11976524196461025, 0.030824947936861986, 1.5640242186704012], [-0.007070007059487528, -0.40606979758147266, 0.48517166367668135, 0.015732334176241397, -0.10365905516273669, 1.5596379227432804], [-0.007093204391756819, -0.5201070602363647, 0.48519422311795257, 0.18999077293139788, -0.27578470610437894, 1.5416818896012996], [-0.09785896211073992, -0.5200986115022112, 0.4851901659702309, 0.000772821799459404, -0.5001490908302251, 1.546414985247032], [-0.09784811292205318, -0.4234946737436865, 0.48514704904918204, -0.061167287502784995, -0.3210308729594724, 1.570000866523097], [-0.09784181396201254, -0.32786856671533054, 0.48519117942010204, -0.22384760106736679, -0.15522927589245705, 1.5991471276301092], [-0.1278482894281499, -0.26330288034124544, 0.4851757032967745, -0.28772349722451523, -0.08943487493120213, 1.6070807069811506], [-0.25049852198401656, -0.27964757562706116, 0.48518943573132883, -0.2877624699826542, -0.08933544421656817, 1.6071518827898517], [-0.25051482323343266, -0.3786747397372159, 0.4851854598711064, -0.29586054537027356, -0.2754981803365959, 1.5842246317083821], [-0.2505022053807022, -0.47303357328725126, 0.46144132731936993, -0.1210125205313525, -0.5329267935306314, 1.5090697910027466], [0.13540092666985082, -0.3528114686870019, 0.5472118128272788, -0.0002327661865520914, -0.00032729046443897653, -0.03465209335916285], [0.1354040726175693, -0.20886963722048973, 0.49749341145936643, -0.2965945310227798, 0.03433197239498291, -0.04026659907726543], [0.13540303850013882, -0.38010470613937114, 0.4974855593500809, -0.15477727957438822, -0.00019740022443600082, -0.04590871653879979], [0.13539472658174426, -0.6044746513885807, 0.41174419378874444, 0.2472218570587093, 0.008962512112892443, -0.04508740821536415]]
        
        # Collect data
        ee_poses = []
        camera_checkerboard_poses = []
        self.rtde_c.moveJ(self.take_photo_config, self.joint_acc_fast, self.joint_vel_fast)
        for i, ee_config in enumerate(calibration_ee_configs):
            self.rtde_c.moveL(ee_config, self.tool_vel, self.tool_acc)
            time.sleep(1)
            ee_pose = self.rtde_r.getActualTCPPose()

            color_img, depth_img = camera.get_data()
            camera_checkerboard_pose = get_camera_checkerboard_pose(
                color_img,
                checkboard_size,
                camera.intrinsics,
                camera.distortion_coeffs,
                obj_points,
                criteria,
            )
            if camera_checkerboard_pose is not None:
                ee_poses.append(list(ee_pose))
                camera_checkerboard_poses.append(camera_checkerboard_pose)
            else:
                print(f"Failed to detect checkerboard in image {i}")

        def compute_calibration_error(ee_poses, camera_checkerboard_poses, cam2gripper):
            transformed_checkerboard_poses = []

            for ee_pose, cam_checkerboard_pose in zip(ee_poses, camera_checkerboard_poses):
                ee_pose_mat = np.eye(4)
                ee_pose_mat[:3, :3] = cv2.Rodrigues(np.array([ee_pose[3:]]))[0]
                ee_pose_mat[:3, 3] = ee_pose[:3]
                cam_checkerboard_pose_mat = np.eye(4)
                cam_checkerboard_pose_mat[:3, :3] = cv2.Rodrigues(np.array([cam_checkerboard_pose[:3]]))[0]
                cam_checkerboard_pose_mat[:3, 3] = cam_checkerboard_pose[3:]

                # Compute the camera pose in the robot base frame
                cam_base_pose = np.dot(ee_pose_mat, cam2gripper)

                # Transform the camera-checkerboard pose to the robot base frame
                transformed_checkerboard_pose = np.dot(cam_base_pose, cam_checkerboard_pose_mat)
                transformed_checkerboard_poses.append(transformed_checkerboard_pose[:, 3][:3])  # Keep only the translation part

            # Calculate the mean and standard deviation of the transformed checkerboard poses
            mean_checkerboard_pose = np.mean(transformed_checkerboard_poses, axis=0)
            print(transformed_checkerboard_poses)
            std_checkerboard_pose = np.std(transformed_checkerboard_poses, axis=0)

            return std_checkerboard_pose

        if len(ee_poses) > 3:
            R_gripper2base = np.array([p[3:] for p in ee_poses])
            t_gripper2base = np.array([p[:3] for p in ee_poses])
            R_target2cam = np.array([p[:3] for p in camera_checkerboard_poses])
            t_target2cam = np.array([p[3:] for p in camera_checkerboard_poses])
            print(R_gripper2base)

            R_cam, T_cam = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_TSAI
            )
            cam_ee_pose = np.eye(4)
            cam_ee_pose[:3, :3] = R_cam
            cam_ee_pose[:3, 3] = T_cam.flatten()

            # Update camera pose
            camera.camera2ee_pose = cam_ee_pose
            camera.camera_pose_inv = np.linalg.inv(camera.camera2ee_pose)

            # Compute calibration error
            std_checkerboard_pose = compute_calibration_error(ee_poses, camera_checkerboard_poses, cam_ee_pose)
            print("Std error: ", std_checkerboard_pose)

            # Save camera optimized offset and camera pose
            print("Saving...")
            np.savetxt("real_camera/camera2ee_pose.txt", camera.camera2ee_pose, delimiter=" ")
            print(camera.camera2ee_pose)
            print("Done.")

            self.go_take_photo()

            return cam_ee_pose
        else:
            return None

    def camera_ee2base(self, camera_ee_pose):
        _ee_pose = self.rtde_r.getActualTCPPose()
        _ee_pose = np.array(_ee_pose)
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = cv2.Rodrigues(_ee_pose[3:])[0]
        ee_pose[:3, 3] = _ee_pose[:3]
        return np.dot(ee_pose, camera_ee_pose)

    def go_drop(self):
        self.rtde_c.moveJ(self.drop_joint_config, self.joint_acc_fast, self.joint_vel_fast)

    def go_pre_grasp(self):
        self.rtde_c.moveJ(self.pre_grasp_config, self.joint_acc_fast, self.joint_vel_fast)

    def go_take_photo(self):
        self.rtde_c.moveJ(self.take_photo_config, self.joint_acc_fast, self.joint_vel_fast)

    def vg_grip(self, timeout=3):
        print("vacuuming...")
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 1)

        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            curr_vg = self.rtde_r.getOutputIntRegister(self.vg_cmd_out_reg)
            if curr_vg >= self.vg_threshold:
                time.sleep(0.5)
                print(f"vacuuming done, current vacuum value = {curr_vg}")
                self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
                return True

        print(f"vacuuming timeout ({timeout}), current vacuum value = {curr_vg}")
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
        time.sleep(0.1)

        return False

    def vg_release(self, timeout=1):
        print("releasing...")
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 2)

        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            curr_vg = self.rtde_r.getOutputIntRegister(self.vg_cmd_out_reg)
            if curr_vg < 1:
                print("releasing done")
                self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
                time.sleep(0.1)
                return True

        print(f"releasing timeout ({timeout}), current vacuum value = {curr_vg}")
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 0)
        time.sleep(0.1)

        return False
    
    def pick_place(self, pick_pose, place_pose):
        pick_pose = [*pick_pose[0:3], 0, 0, pick_pose[3] + self.take_photo_yaw]
        place_pose = [*place_pose[0:3], 0, 0, place_pose[3] + self.take_photo_yaw]

        # prepare pick pose
        pre_pick1 = pick_pose.copy()
        pre_pick1[2] += 0.1
        pre_pick2 = pick_pose.copy()
        pre_pick2[2] += 0.02
        pick_pose[2] -= 0.03

        # move to pre_pick1
        joints = self.rtde_c.getInverseKinematics(pre_pick1, qnear=self.pre_grasp_config)
        self.rtde_c.moveJ(joints, self.joint_vel_fast, self.joint_acc_fast)
        # self.rtde_c.moveL(pre_pick1, self.tool_vel, self.tool_acc)
        # move to pre_pick2
        self.rtde_c.moveL(pre_pick2, self.tool_vel, self.tool_acc)
        # contact the object
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 1)
        self.rtde_c.moveL(pick_pose, self.tool_vel_slow, self.tool_acc_slow, asynchronous=True)
        while not all(np.isclose(self.rtde_r.getActualTCPPose()[0:3], pick_pose[0:3], atol=0.002)):
            force = self.rtde_r.getActualTCPForce()
            curr_vg = self.rtde_r.getOutputIntRegister(self.vg_cmd_out_reg)
            if (np.abs(force[2])) > self.force_threshold or curr_vg >= self.vg_threshold:
                print("force:", force, "curr_vg:", curr_vg)
                self.rtde_c.stopL(self.tool_acc_stop, False)
                break
        # grip
        success = self.vg_grip()
        if not success:
            return False

        # update place pose
        object_z = self.rtde_r.getActualTCPPose()[2]
        after_pick = pick_pose.copy()
        after_pick[2] = object_z + 0.15
        pre_place = place_pose.copy()
        pre_place[2] = object_z + 0.15
        place_pose[2] = object_z + 0.002
        after_place = place_pose.copy()
        after_place[2] += 0.05

        # move after pick
        self.rtde_c.moveL(after_pick, self.tool_vel, self.tool_acc)
        # move to pre_place
        joints = self.rtde_c.getInverseKinematics(pre_place, qnear=self.pre_grasp_config)
        self.rtde_c.moveJ(joints, self.joint_vel_fast, self.joint_acc_fast)
        # self.rtde_c.moveL(pre_place, self.tool_vel, self.tool_acc)
        # move down to place pose
        self.rtde_c.moveL(place_pose, self.tool_vel, self.tool_acc)
        # release
        self.vg_release()
        time.sleep(0.5)
        # move up to after_place
        # joints = self.rtde_c.getInverseKinematics(after_place)
        # joints[-1] = np.radians(-19)
        # self.rtde_c.moveJ(joints, self.joint_vel_fast, self.joint_acc_fast)
        self.rtde_c.moveL(after_place, self.tool_vel, self.tool_acc)
        
        
        # self.go_take_photo()

        return True
    
    def pick_drag(self, path, height):
        pick_pose = [path[0][0], path[0][1], height, 0, 0, path[0][2] + self.take_photo_yaw]
        
        # prepare pick pose
        pre_pick1 = pick_pose.copy()
        pre_pick1[2] += 0.1
        pre_pick2 = pick_pose.copy()
        pre_pick2[2] += 0.02
        pick_pose[2] -= 0.03
        
        # move to pre_pick1
        joints = self.rtde_c.getInverseKinematics(pre_pick1, qnear=[np.radians(69), np.radians(-111), np.radians(116), np.radians(-94), np.radians(-89), np.radians(-23)])
        self.rtde_c.moveJ(joints, self.joint_vel_fast, self.joint_acc_fast)
        # self.rtde_c.moveL(pre_pick1, self.tool_vel, self.tool_acc)
        # move to pre_pick2
        self.rtde_c.moveL(pre_pick2, self.tool_vel, self.tool_acc)
        # contact the object
        self.rtde_i.setInputIntRegister(self.vg_cmd_in_reg, 1)
        self.rtde_c.moveL(pick_pose, self.tool_vel_slow, self.tool_acc_slow, asynchronous=True)
        while not all(np.isclose(self.rtde_r.getActualTCPPose()[0:3], pick_pose[0:3], atol=0.002)):
            force = self.rtde_r.getActualTCPForce()
            curr_vg = self.rtde_r.getOutputIntRegister(self.vg_cmd_out_reg)
            if (np.abs(force[2])) > self.force_threshold or curr_vg >= self.vg_threshold:
                print("force:", force, "curr_vg:", curr_vg)
                self.rtde_c.stopL(self.tool_acc_stop, False)
                break
        # grip
        success = self.vg_grip()
        if not success:
            return False

        # update place pose
        object_z = self.rtde_r.getActualTCPPose()[2]
        after_pick = pick_pose.copy()
        after_pick[2] = object_z + 0.001
        
        # move after pick
        self.rtde_c.moveL(after_pick, self.tool_vel, self.tool_acc)
        # follow the path
        for pose in path:
            self.rtde_c.moveL([*pose[0:2], after_pick[2], 0, 0, pose[2] + self.take_photo_yaw], self.tool_vel_fast, self.tool_acc_fast)
        # TODO: fix this
        # tcp_path = []
        # blend = 0.0
        # for pose in path:
        #     tcp_path.append([*pose[0:2], after_pick[2], 0, 0, pose[2] + self.take_photo_yaw, self.tool_vel, self.tool_acc, blend])
        # self.rtde_c.moveL(tcp_path)
        # self.rtde_c.stopScript()
        # release
        after_place = [path[-1][0], path[-1][1], after_pick[2]+ 0.05, 0, 0, path[-1][2] + self.take_photo_yaw]
        self.vg_release()
        time.sleep(0.5)
        # move up to after_place
        # joints = self.rtde_c.getInverseKinematics(after_place)
        # joints[-1] = np.radians(-19)
        # self.rtde_c.moveJ(joints, self.joint_vel_fast, self.joint_acc_fast)
        self.rtde_c.moveL(after_place, self.tool_vel, self.tool_acc)

        # self.go_take_photo()

        return True

if __name__ == "__main__":
    robot = UR5_VGA10()
    init_q = robot.get_joint_state()
    print("Initial joint state:", init_q)
    init_q[0]  += 0.1
    robot.moveJ(init_q, 0.5, 0.5)
    init_q[0]  -= 0.1
    robot.moveJ(init_q, 0.5, 0.5)

    # from rtde_control import Path
    # velocity = 0.5
    # acceleration = 0.5
    # blend_1 = 0.0
    # blend_2 = 0.02
    # blend_3 = 0.0
    # path_pose1 = [-0.143, -0.435, 0.20, -0.001, 0, 0.04, velocity, acceleration, blend_1]
    # path_pose2 = [-0.143, -0.51, 0.21, -0.001, 0, 0.04, velocity, acceleration, blend_2]
    # path_pose3 = [-0.32, -0.61, 0.31, -0.001, 0, 0.04, velocity, acceleration, blend_3]
    # path = [path_pose1, path_pose2, path_pose3]
    # new_path = Path()
    # new_path.appendMovelPath(path)
    # robot.rtde_c.movePath(new_path)
    # # robot.rtde_c.stopScript()
    # exit()

    # robot.vg_grip()

    # input('wait')

    # robot.vg_release()

    # from real_camera.camera import Camera

    # camera = Camera()

    # # robot.calibrate_marker(camera, False)

    # robot.handeye_calibration(camera)

    # robot.rtde_c.stopScript()
