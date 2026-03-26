from setuptools import find_packages, setup

package_name = 'kc_3578'

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
    maintainer='ashish',
    maintainer_email='ashishjha1433@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "arm_manipulation = kc_3578.arm_manipulation:main",
            "perception = kc_3578.perception:main",
            "navigation = kc_3578.navigation:main",
            "shape_detection = kc_3578.shape_detection:main",
        ],
    },
)
