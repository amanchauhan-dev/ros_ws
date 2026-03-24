from setuptools import find_packages, setup

package_name = 'task5_3578'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'scipy'
    ],
    zip_safe=True,
    maintainer='aman',
    maintainer_email='amanchauhan0435@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # nav scripts
            # V1
            "nav = task5_3578.nav.nav:main",
            "shape = task5_3578.nav.shape:main",
            "shape_v1 = task5_3578.nav.shape_v1:main",
            "shape_viz = task5_3578.nav.shape_viz:main",
            # V2
            "nav_2 = task5_3578.nav_2.nav:main",
            "shape_2 = task5_3578.nav_2.shape:main",
            "shape_v2 = task5_3578.nav_2.shape_v1:main",
            # arm scripts
            "fi = task5_3578.arm.ferti_init:main",
            "fus = task5_3578.arm.ferti_using_servo:main",
            "p = task5_3578.arm.prescription:main",
            "tmcr = task5_3578.arm.task4a_motion_check_real:main",
            "t = task5_3578.arm.task4A:main",
            "tpr = task5_3578.arm.task4c_pre_run:main",
        ],
    },
)
