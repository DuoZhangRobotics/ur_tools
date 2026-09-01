from setuptools import find_packages, setup

setup(
    name='ur_tools',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'natnet==0.3.0',
        'numpy>=1.26,<2',
        'opencv-python>=4.10,<4.12',
        'pyrealsense2',
        'PyYAML>=6',
        'scipy>=1.11,<2',
        'ur_rtde',
    ],
    extras_require={'test': ['pytest>=8,<9']},
    entry_points={
        'console_scripts': [
            'ur-optitrack-calibrate=ur_tools.optitrack.cli:main',
            'ur-optitrack-coverage=ur_tools.optitrack.coverage_cli:main',
            'ur-optitrack-publisher=ur_tools.optitrack.natnet_pose_publisher:main',
        ],
    },
    author='Duo Zhang, Baichuan Huang, Kowndinya Boyalakuntla',
    description='UR5 robot calibration and control tools with RealSense cameras and Robotiq 2f 85 gripper and OnRobot VGA10 gripper',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DuoZhangRobotics/RobotControl.git',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
