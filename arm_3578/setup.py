from setuptools import find_packages, setup

package_name = 'arm_3578'

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
            "ff = arm_3578.final.find_final:main",
            "fs = arm_3578.final.fruits_speed_sort:main",
            "sv = arm_3578.final.speed_version_final_fruits:main",
            "us = arm_3578.final.unload_script:main",

            # Real case
            "fr = arm_3578.real_cases.fruits_real:main",
            "r1 = arm_3578.real_cases.real_case1:main",
            "r2 = arm_3578.real_cases.real_case2:main",
            "r3 = arm_3578.real_cases.real_case3:main",
            "r5 = arm_3578.real_cases.real_case5:main",
        ],
    },
)
