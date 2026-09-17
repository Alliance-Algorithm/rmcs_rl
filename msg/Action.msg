std_msgs/Header header
uint64 obs_seq                  # 回显所应答的 Observation.obs_seq
uint64 layout_hash              # 回显策略进程由模型 metadata 算出的合同指纹
uint64 model_id                 # FNV-1a64(模型文件字节)：桥据此对账"跑的是不是这份权重"
float64[] action                # 长度 == action_size；原始 actor 输出
