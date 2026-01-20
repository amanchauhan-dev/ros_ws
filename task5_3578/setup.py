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
            "nav = task5_3578.nav.nav:main",
            "shape = task5_3578.nav.shape:main",
            "shape_v1 = task5_3578.nav.shape_v1:main",
            # raghav
            "fi = task5_3578.ferti_init:main",
            "fus = task5_3578.ferti_using_servo:main",
            "p = task5_3578.prescription:main",
            "tmcr = task5_3578.task4a_motion_check_real:main",
            "t = task5_3578.task4A:main",
            "tpr = task5_3578.task4c_pre_run:main",
        ],
    },
)
