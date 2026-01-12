from setuptools import find_packages, setup

package_name = 'task4b_3578'

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
        "numpy"
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
            # shape
            "shape_final_v1 = task4b_3578.shape_final_v1:main",
            # navigation
            "nav_final_v1 = task4b_3578.nav_final_v1:main",
            "nav_final_v2 = task4b_3578.nav_final_v2:main",
        ],
    },
)
