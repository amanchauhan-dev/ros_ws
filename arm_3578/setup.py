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
            "va = arm_3578.final.ferti_unload_v3A:main",
            "vb = arm_3578.final.ferti_unload_v3B:main",



            # Real case
            "fr = arm_3578.real_cases.fruits_real:main",
            "r1 = arm_3578.real_cases.real_case1:main",
            "r2 = arm_3578.real_cases.real_case2:main",
            "r3 = arm_3578.real_cases.real_case3:main",
            "r5 = arm_3578.real_cases.real_case5:main",

            # New
            "fam = arm_3578.new.final_arm_manipulation_3578:main",
            "fv2 = arm_3578.new.final_v2:main",
            "hbl = arm_3578.new.hybrid_logic:main",
            "ffs = arm_3578.new.final_ferti_sol:main",
            "pr = arm_3578.new.pr:main",
            "ffsc = arm_3578.new.fianl_fert_sol_change:main",
        ],
    },
)
