std_msgs/Header header
string model_name               # 模型文件名
uint64 model_id                 # FNV-1a64(模型文件字节)
string policy_version           # 模型 metadata 的 policy_version（可空）
uint64 layout_hash              # 由模型 metadata 算出的合同指纹
uint32 obs_size
uint32 action_size
float64 inference_p50_us
float64 inference_p99_us
