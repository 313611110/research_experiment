#!/usr/bin/env python3
import rospy
import os
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest
from voronoi_cbsa.msg import TargetInfoArray  # 根据你实际的消息定义调整导入

def spawn_target(target):
    """
    根据目标信息调用 Gazebo spawn 服务生成模型
    """
    model_name = "target_{}".format(target.id)
    # 设置目标初始位姿，这里将 z 设为 0.1 以防模型被地面遮挡
    initial_pose = Pose(
        position=Point(target.position.x, target.position.y, 0.1),
        orientation=Quaternion(0, 0, 0, 1)
    )

    # 获取模型文件路径，这里假定 target_spawner.py 与 models 目录在同一包下
    package_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(package_path, "..", "models", "target_box", "model.sdf")
    try:
        with open(model_path, 'r') as f:
            model_xml = f.read()
    except Exception as e:
        rospy.logerr("Failed to read model file: {}".format(e))
        return

    rospy.loginfo("Spawning target model: {}".format(model_name))
    # 等待 spawn 服务
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        req = SpawnModelRequest()
        req.model_name = model_name
        req.model_xml = model_xml
        req.robot_namespace = rospy.get_namespace()  # 或者设定为特定 namespace
        req.initial_pose = initial_pose
        req.reference_frame = "world"
        resp = spawn_model(req)
        rospy.loginfo("Spawn status: {}".format(resp.status_message))
    except Exception as e:
        rospy.logerr("Spawn service call failed: {}".format(e))

def target_callback(msg):
    """
    回调函数：当收到目标信息时，逐个生成目标模型
    """
    # 这里示例仅对每个目标生成一次模型。如果目标动态变化后需要更新位置，
    # 可考虑调用 /gazebo/set_model_state 服务更新模型状态
    for target in msg.targets:
        spawn_target(target)

if __name__ == '__main__':
    rospy.init_node('target_spawner', anonymous=True)
    rospy.Subscriber("/target", TargetInfoArray, target_callback)
    rospy.loginfo("Target spawner node started, waiting for /target messages...")
    rospy.spin()
