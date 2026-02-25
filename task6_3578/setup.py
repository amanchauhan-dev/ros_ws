from setuptools import find_packages, setup

package_name = 'task6_3578'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
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
            # controllers
            "cxn = task6_3578.controller.control_XN:main",
            "cxp = task6_3578.controller.control_XP:main",
            "cyn = task6_3578.controller.control_YN:main",
            "cyp = task6_3578.controller.control_YP:main",
            "czn = task6_3578.controller.control_ZN:main",
            "czp = task6_3578.controller.control_ZP:main",
            # do or die
            "rv1 = task6_3578.do_or_die.rdd_v1:main",
            "rv2 = task6_3578.do_or_die.rdd_v2:main",
            # fruits dustbin sort
            "fv1 = task6_3578.fruits_dustbin_sort.fds_v1:main",
            "fv2 = task6_3578.fruits_dustbin_sort.fds_v2:main",
        ],
    },
)
