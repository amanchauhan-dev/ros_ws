from setuptools import find_packages, setup

package_name = 'nav_kc3578'

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
            # Simulation Virtual World Nodes
            "sim_nav = nav_kc3578.sim_nav:main",
            "sim_line_follow = nav_kc3578.sim_line_follow:main",
            "sim_shape = nav_kc3578.sim_shape:main",

            # Real Virtual World Nodes
            "real_shape = nav_kc3578.real.real_shape:main",
            "real_shape_v1 = nav_kc3578.real.real_shape_v1:main"
        ],
    },
)
