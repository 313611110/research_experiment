#!/usr/bin/env python3
import rospy
import pandas as pd
from geometry_msgs.msg import Pose
from std_msgs.msg import Int16
import os

class AgentPositionLogger:
    def __init__(self):
        rospy.init_node('agent_pos_csv', anonymous=True)
        self.agent_positions = {i: {"x": 0.0, "y": 0.0} for i in range(6)}  # 儲存6個agent的位置
        self.frame_id = 0
        self.data = []

        # 訂閱每個agent的位置
        for i in range(6):
            rospy.Subscriber(f"/agent_{i}/visualize/pose", Pose, self.pose_callback, callback_args=i)

        # 定時保存資料到CSV
        self.save_rate = rospy.Rate(1)  # 每秒更新一次
        self.save_path = os.path.expanduser("~/research_ws/src/voronoi_cbsa/scripts/Voronoi_based_controller/Ver3/agent_pos.csv")

    def pose_callback(self, msg, agent_id):
        self.agent_positions[agent_id]["x"] = msg.position.x
        self.agent_positions[agent_id]["y"] = msg.position.y

    def run(self):
        while not rospy.is_shutdown():
            # 儲存每個frame的資料
            row = {"frame_id": self.frame_id}
            for i in range(6):
                row[f"agent_{i}_x"] = self.agent_positions[i]["x"]
                row[f"agent_{i}_y"] = self.agent_positions[i]["y"]
            self.data.append(row)

            # 每10個frame儲存到CSV
            if self.frame_id % 10 == 0:
                df = pd.DataFrame(self.data)
                df.to_csv(self.save_path, index=False)
                rospy.loginfo(f"Data saved to {self.save_path}")

            self.frame_id += 1
            self.save_rate.sleep()

if __name__ == "__main__":
    logger = AgentPositionLogger()
    try:
        logger.run()
    except rospy.ROSInterruptException:
        pass
