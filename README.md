# Workspace *ros_ws*

## *RULES*
1. All the *Packages* should be at root level.
2. For testing codes which you dont want to put in package, you can use **1_not_pkg**. In this directory you can put any file any test code, jupyter files, etc. KEEP IN MIND IT'S NOT A PACKAGE
3. Your packages should be independent of VENV or CONDA. 


## Directory *(IMPORTANT)*

- **1_not_pkg**: 
    - This directory contains all the test codes. 
    - *NOTE* "This is not node"
- [...rest]: This all are nodes

# FOR 1_not_pkg 

1. Create a venv 
    - *NOTE: DON'T CREATE THIS VENV INSIDE ANY WORKSPACE*
2. Run:
    ```bash
    pip3 install -r requirements.txt
    ```
- Select python interpreter as your venv

## NOTE: requirements.txt is not required to run nodes


# FOR ROS NODES

- venv not required
- Build your workspace 
    ```bash
    colcon build
    ```

- Source your node
    ```bash
    source install/setup.bash
    ```
- Run your nodes
    ```bash
    ros2 run <pkg_name> <node_name>
    ```
