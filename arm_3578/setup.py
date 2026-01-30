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
            "pv1 = arm_3578.precption.preception_new_v1:main",
            "pv2 = arm_3578.precption.fruit_broadcaster:main",
            "tmr = arm_3578.task5_ferti_m_real:main"
        ],
    },
)
