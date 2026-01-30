from setuptools import find_packages, setup

package_name = 'ebot_3578'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy',],
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
            "nav = ebot_3578.nav:main",
            "shape = ebot_3578.shape:main",
            "shape_v1 = ebot_3578.shape_v1:main",
            "shape_viz = ebot_3578.shape_viz:main",
        ],
    },
)
