std_msgs/Header header          # stamp = 采样时刻（桥填 ROS 时间）；frame_id 留空
uint64 obs_seq                  # 单调递增的观测序号，桥生成
uint64 layout_hash              # 桥按 YAML 词条算出的合同指纹（FNV-1a64，见 planning/docs/bridge-design.md §6.2）
float64[] obs                   # 长度 == obs_size
