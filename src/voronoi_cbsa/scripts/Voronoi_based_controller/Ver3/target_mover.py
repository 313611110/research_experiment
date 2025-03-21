#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from voronoi_cbsa.msg import TargetInfoArray

class TargetMover:
    def __init__(self):
        rospy.init_node("target_mover", anonymous=True)
        self.model_name = rospy.get_param("~model_name", "target_0")  # 從 launch 檔案讀取目標名稱
        self.set_state_service = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        rospy.wait_for_service("/gazebo/set_model_state")

        # 訂閱 `/target` topic，獲取目標的動態位置
        rospy.Subscriber("/target", TargetInfoArray, self.target_callback)
        rospy.loginfo("Target Mover Node Started")

    def target_callback(self, msg):
        for target in msg.targets:
            if target.id == int(self.model_name.split("_")[-1]):  # 確保這個節點控制正確的 Target
                state_msg = ModelState()
                state_msg.model_name = self.model_name
                state_msg.pose.position.x = target.position.x
                state_msg.pose.position.y = target.position.y
                state_msg.pose.position.z = 0

                # 速度控制 (非必要，可以讓 Gazebo 自然模擬)
                state_msg.twist.linear.x = target.velocity.linear.x
                state_msg.twist.linear.y = target.velocity.linear.y
                state_msg.twist.linear.z = 0

                try:
                    self.set_state_service(state_msg)
                except rospy.ServiceException as e:
                    rospy.logerr("Failed to update target position: %s" % e)

if __name__ == "__main__":
    TargetMover()
    rospy.spin()
