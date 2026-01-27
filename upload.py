from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'ms-ef223cb6-8146-49e8-96f4-ba6231fd3e26'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

api.upload_folder(
    repo_id=f"canalpang/crossnd-whoiswho",
    folder_path='./whoiswho_data',
    commit_message='upload dataset folder to repo',
    repo_type = 'dataset'
)