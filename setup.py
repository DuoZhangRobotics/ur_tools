from setuptools import setup, find_packages

setup(
    name='ur_tools',
    version='0.1.0',
    packages=find_packages("ur_tools", "ur_tools.*"),
    install_requires=[
        'opencv-python',
        'numpy',
        'pyrealsense2',
        'scipy',
        'ur_rtde'
        # Add any other dependencies
    ],
    author='Duo Zhang, Baichuan Huang, Kowndinya Boyalakuntla',
    description='UR5 robot calibration and control tools with RealSense cameras and Robotiq 2f 85 gripper and OnRobot VGA10 gripper',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DuoZhangRobotics/RobotControl.git',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
