import os
from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset

# 设置下载路径
download_dir = './whoiswho_data'
os.makedirs(download_dir, exist_ok=True)

# 验证 ModelScope token
api = HubApi()
api.login('ms-5ef9d8f4-c656-48a4-9af9-b6e660d4ee42')

# 数据集下载
ds = MsDataset.load('canalpang/crossnd-whoiswho', cache_dir=download_dir)
