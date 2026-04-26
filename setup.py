from setuptools import setup

package_name = 'pseudo_rgbd_slam'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='developer@example.com',
    description='Pseudo RGB-D SLAM with neural depth estimation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'node_a = pseudo_rgbd_slam.node_a_broadcaster:main',
            'node_b = pseudo_rgbd_slam.node_b_depth_estimator:main',
        ],
    },
)
