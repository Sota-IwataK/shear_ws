from setuptools import setup

setup(
    name='share_semantic_transport', version='0.1.0',
    packages=['share_semantic_transport'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/share_semantic_transport']),
        ('share/share_semantic_transport', ['package.xml']),
    ],
    install_requires=['setuptools'], tests_require=['pytest'],
    maintainer='iwata', maintainer_email='your_email@example.com', license='MIT',
    description='Hardware-independent semantic transport',
    entry_points={'console_scripts': [
        'local_hold_semantic_mock = share_semantic_transport.node:main',
    ]},
)
