# Workspace *ros_ws*

# FOR test_codes 

- Create a venv
- Run 

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
